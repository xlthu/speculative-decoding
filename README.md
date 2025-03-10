# 投机推理

实现算法：
- 自回归（无投机）
- Prompt Lookup Decoding（PLD）
    - https://github.com/apoorvumang/prompt-lookup-decoding
- Token Recycling
    - https://arxiv.org/abs/2408.08696

问题：
- Token Recycling 方法使用原文中的树或 Eagle-1 的树
    - 在 Mac CPU/GPU 平台上输出均与非投机版本存在较大差异
    - 不确定代码存在 BUG 还是浮点误差导致
