---
layout:     post
title:      "量化OP实现——QuantizedAdd"
subtitle:   ""
date:       2018-05-17
author:     "dustless"
header-img: ""
catalog: true
tags:
    - 量化
    - tensorflow
    - nnlib
---


## 量化op

一般而言，量化后的op，输入为一组或多组[input(uint8),  input_min(float),  input_max(float)]，输出为一组或多组[output(uint8/int32), output_min(float), output_max(float)]。

量化后的op实现的要点在于，如何使用量化后的定点数值进行相应的运算（尽量避免批量的浮点数运算，否则在DSP这种平台上无法保证性能），得到最终结果的近似量化值。

本文以QuantizedAdd为例，说明量化op的一些实现要点。

## 定点计算的基本操作
考虑输入一个已经量化好的数组x（每个元素范围0~255的8bit整数），现在需要给这组输入乘上一个浮点系数a，如何用整数乘法来实现？

1. 选取适当精度，将浮点系数a表示成定点整数，如ia = round(a * 128.0)
2. for each x, 计算y = (x * ia) / 128， 其中除法运算可以用移位操作来实现。

这样计算的精度是否足够呢？首先第一步的转换操作引入的系数误差最大为0.5，这样在第二步的乘法+移位操作之后，输出的误差最大为255 * 0.5 / 128 < 1, 这样的误差还是能接受的。如果需要更高的精度，可以在第一步转换系数的时候乘上更大的系数，相应的第二步的移位操作就需要移位更多位。

## QuantizedAdd8p8to32

输入：uint8 *a, float a_min, float a_max, uint8 *b, float b_min, float b_max

输出：int32 *c, c_min, c_max

#### 1. Tensorflow QuantizedAdd的参考实现

如何选定输出的范围？
* 输出是有符号的
* 输出的范围必须关于0点对称，这样才能使得0+0=0
* 能包含输入的最大范围，尽量大的范围可以保证不溢出
* 应该有足够的精度，如果可能的话，没有更低的位被截断
* 高位应留有空间，防止溢出

输入是8bit，输出是32bit, 令
```
c_max = max(a_max, -a_min, b_max, -b_min) * 2^17
c_min = -c_max
```
这样就可用高位17bit（1bit符号位+16bit值）保证足够大的范围，低位15bit保证精度。

output_value计算过程：
```
// 计算0.0在total_space中的表示
int32 zero_in_total_space = FloatToQuantized<int32>(0.0f, c_min, c_max)
// 将输入转为float，再在total_space中量化成int32表示
int32 a_in_total_space = RequantizeInNewRange<uint8, int32>(a_value, a_min, a_max, c_min, c_max)
int32 b_in_total_space = RequantizeInNewRange<uint8, int32>(b_value, b_min, b_max, c_min, c_max)
// 最后的输出值还需要加上实际0点的偏移量，一般情况下zero_in_total_space=0
int32 c_value = a_in_total_space + b_in_total_space + zero_in_total_space
```

#### 2. nnlib的实现

Tensorflow中QuantizedAdd的实现包含了两次Requantize操作，用hvx实现的话，可以考虑将其展开。参考代码如下：

```
void add8p8to32(uint8_t *a, float a_max, float a_min,
                uint8_t *b, float b_min, float b_max,
                uint8_t *c, float *c_min, float *c_max,
                int length) {
  float a_level_size = (a_max - a_min) / 255;
  float b_level_size = (b_max - b_min) / 255;

  *c_max = fmaxf(fabsf(a_min), a_max);
  *c_max = fmaxf(*c_max, fmaxf(fabsf(b_min), b_max));
  *c_max *= (1 << 17);
  *c_min = -(*c_max);
  float c_level_size = (c_max - c_min) / 4294967296.0f/*0x1.0p32f*/;

  uint8_t a_sub_amt = quantize_uint8(0.0f, a_min, a_max);
  uint8_t b_sub_amt = quantize_uint(0.0f, b_min, b_max);

  float a_mpy_amt = (a_level_size / c_level_size);  // <2**7
  float b_mpy_amt = (b_level_size / c_level_size);  // <2**7
  
  // 此处为ref版本实现，可替换成hvx实现
  for (int i = 0; i < length; ++i) {
    c[i] = (int32_t) (0.5f + ((a[i] - a_sub_amt) * a_mpy_amt)
        + (((int32_t) b[i] - b_sub_amt) * b_mpy_amt));
  }
}

```

## QuantizedAdd8p8to8

输入：uint8 *aq, float amin, float amax, uint8 *bq, float bmin, float bmax, float gmin, float gmax

输出：uint8 *cq, cmin, cmax

这个op相当于是将QuantizedAdd8p8to32和Requantize32to8合并成了一个op。其中[gmin, gmax]表示**猜测**的输出范围。合并这两个操作之后可以省去中间结果（32bit）的存储和load开销，降低运算复杂度。

**备注**：猜测的范围是如何work的？基于猜测的范围可以计算出输出cq，判断cq是否产生了溢出，如果产生了溢出，那么调整范围，再计算一次，按照一定的策略，基本上就能保证只需要“猜”两次就能得到一个相对合理的结果。

贴上nnlib中的实现：
```
static inline void do_quantized_add_888(uint8_t *aq,
                                        float amax,
                                        float amin,
                                        uint8_t *bq,
                                        float bmax,
                                        float bmin,
                                        float gmax,
                                        float gmin,
                                        uint8_t *cq,
                                        float *cmax,
                                        float *cmin,
                                        int length,
                                        int16_t *scratch) {
  float arange = amax - amin;
  float brange = bmax - bmin;
  float step, lmin, lmax;
  float alpha = arange / brange;
  if (alpha >= 256.0f) {
    vmemcpy_asm(cq, aq, length);
    *cmax = amax;
    *cmin = amin;
    return;
  }
  int16_t *ptr_max = scratch;
  short ialpha = 128.0f * alpha;
  float kappa = 128.0f * alpha + (255.0f * amin + 255.0f * bmin) / stepb;
  short ikappa = (int) (kappa + .0f); //+ialpha is because input is 08 ^
  //compute local max,min by updating local
  lmin = (gmin * 255.0f) / stepb;
  lmax = (gmax * 255.0f) / stepb;
  step = lmax - lmin;
  float frecip = (255.0f * 32768.0f) / step;
  float foffset = (255.0f * lmin) / step;
  if (frecip >= 32767.0f) frecip = 32767.0f;
  short recip = (int) (frecip + 0.0f);
  short offset = (int) (foffset - 0.5f);
  //printf("frecip=%f foffset=%f recip=%x offset=%x step=%f arange=%f brange=%f gmax=%f gmin=%f alpha=%f kappa=%f\n",frecip,foffset,recip,offset,step,arange,brange,gmax,gmin,alpha,kappa);
  quant_add_spec_asm(aq,
                     bq,
                     ialpha,
                     ikappa,
                     offset,
                     recip,
                     cq,
                     ptr_max,
                     length);
  float xmax = (float) ptr_max[0];
  float xmin = (float) ptr_max[64];
  //turn back to global max
  *cmin = (xmin * brange) / 255.0f;
  *cmax = (xmax * brange) / 255.0f;
}
```

如何理解上面的代码呢？我们先假设输出的范围就是[gmin, gmax]，于是可以推导出cq的计算表达式。

```math
cfloat = afloat + bfloat;

=> cq*(gmax-gmin)/255 + gmin = aq*arange/255 + amin + bq*brange/255 + bmin

=> cq = [aq*arange/brange + bq + 255*(amin + bmin)/brange] * brange/(gmax-gmin) - 255*gmin/(gmax-gmin)

= [(aq-128)*ialpha/128 + bq + ikappa] * recip/32768 - offset;

```
其中ialpha，kappa，frecip和offset的取值，如代码所述：
```c
alpha = arange / brange;
ialpha = 128.0f * alpha;
ikappa = 128.0f * alpha + (255.0f * amin + 255.0f * bmin) / brange;
recip = 32768 * brange / (gmax - gmin);
offset = 255 * gmin / (gmax - gmin)
```
引入ialpha(16bit)、ikappa(16bit)、recip(16bit)、offset(16bit)等中间变量的目的，是为了更好地进行整数的向量化计算。

那么，如果gmax和gmin不正确，如何才能求出正确的cmax和cmin呢？

注意到```[(aq-128)*ialpha/128 + bq + ikappa]```这一项与猜测的输入gmax和gmin无关，它的值只和真实输入相关，令其为x(16bit整数)，则.
```math
cq = x * recip / 32768 - offset

=> x = (cq + offset) * 32768/recip
= [cq + 255 * gmin / (gmax - gmin)] * (gmax - gmin) / brange
```

理论上，最终计算得到的cq需要在[0, 255]的范围内，所以可以通过第一次计算过程中得到的xmin和xmax来更新gmin和gmax（即cmin和cmax）。

令cq = 0，得到xmin = gmin * 255 / brange => cmin = xmin * brange / 255.

令cq = 255, 得到xmax = gmax * 255 / brange => cmax = xmax * brange / 255.

得到了正确的cmin和cmax后，只需要再调用一次do_quantized_add_888函数即可得到正确的结果。

Quantized8p8to8的实现利用了一点小trick，从而实现了至多两次计算之后就能得到正确的结果。使用Quantized8p8to8操作虽然看起来好像比QuantizedAdd8p8to32+Requantize32to8更麻烦（重复计算），但实际上却节省了中间的存储开销并减少了requantize的运算，性能可以得到很大的提升。


## 参考链接
1. https://github.com/tensorflow/tensorflow
2. https://source.codeaurora.org/quic/hexagon_nn/nnlib
