<!--
 * @Author             : 陈蔚 (weichen.cw@zju.edu.cn)
 * @Date               : 2025-07-21 22:25
 * @Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
 * @Last Modified Date : 2025-07-21 22:25
 * @Description        : 
 * -------- 
 * Copyright (c) 2025 Wei Chen. 
-->

# rStar 算法复现

**题目**：使用pytorch与vllm实现[rStar](https://arxiv.org/abs/2408.06195)基本算法，要求能够运行在gsm8k数据集上的测试。

**算法描述**：rStar算法将模型在数学任务上的解答建模为马科尔夫过程，并使用MCTS算法来生成对问题的解答。马尔可夫过程中的各元素定义具体如下：

* 状态：当前推理的步骤. 
  * 初始状态为原题目；
  * 结束状态为得到原题目的解答。
* 动作：从当前状态到下一个状态所需要执行的动作，包括：
  * A1: One-step thought：直接生成下一步推理步骤；
  * A2: Direct answer：直接生成若干条推理步骤直至给出原题目的答案；
  * A3: Subquestion & Answer：生成一个子问题，再给出子问题的答案；
  * A4: Re-answer subquestion：重新回答上一个子问题；
  * A5: Rephrase question：对原问题进行解释与改写。
备注：动作间的执行顺序有要求，A4只能紧跟着A3或A5执行；A5只能跟在初始状态后执行。
* 奖励函数：采用基于最终答案正确性的奖励回溯机制：
  * 终止节点（叶子结点）的奖励$Q(s_d, a_d)$计算：使用self-consistency多数投票的置信度作为奖励值.
  * 中间节点初始$Q(s)$值设为0;
  * 奖励回溯机制：当到达终端节点后，将奖励值沿轨迹回溯更新所有中间节点的Q值：
    * 更新公式：$Q(s_i, a_i) = Q(s_i, a_i) + Q(s_d, a_d)$
  * 
* MCTS流程：
  * 选择(Selection)：使用UCT算法平衡探索与利用;
  * UCT公式：$\text{UCT}(s,a) = \frac{Q(s,a)}{N(s,a)} + c*\sqrt{\frac{ln(N_\text{parent}(s))}{N(s,a)}}$
    * 其中c是探索系数(建议默认值1.4)
    * $N(s,a)$是节点访问计数
  * 扩展(Expansion)：当遇到未完全展开的节点时扩展新节点;
  * 模拟(Simulation)：在树上重复执行选择和拓展，直到选择到的节点为叶子结点停止，记为一次rollout；
  * 回溯(Backpropagation)：将终端节点奖励回溯更新路径上所有节点.

**输入输出要求**：

* 使用vLLM进行高效的LLM推理，并实现完整的MCTS树结构及搜索过程，支持GSM8K数据集加载和评估。
* 包含以下关键文件：
  * main.py: 包含数据集读取和处理，调用MCTS搜索，生成搜索树以及最终答案并进行保存；
  * MCTS.py: 封装MCTS搜索过程，包含搜索树节点的生成和选择，以及回溯更新。
  * vllm_API: 封装vLLM对模型的调用过程，并提供API接口；
  * eval.py: 评估生成答案的准确率；
  * README.md: 描述项目结构、使用方法、依赖项、注意事项等。
