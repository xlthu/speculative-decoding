# 投机推理

实现算法：
- 自回归（无投机）
- Prompt Lookup Decoding（PLD）
    - https://github.com/apoorvumang/prompt-lookup-decoding
- Token Recycling
    - https://arxiv.org/abs/2408.08696

问题：
- 使用 Token Recycling 原文中的树
    - 在 Mac CPU 平台上输出不通顺，不确定代码存在 BUG 还是浮点误差导致
    - 在 Mac GPU 平台上输出与非投机版本存在较大差异
