# run_repr_demo.py  （原 run.py）
from gp_retro_repr import (
    Program, Select, ApplyTemplate, Stop, ExecutionConfig
)

from demo_utils import build_world_t1  # 统一世界

def main():
    # 1) 统一的库存 + 模板 + 目标
    stock, reg, target = build_world_t1()

    # 2) DP 程序：告诉解释器“怎么拆”
    prog = Program([
        Select(0),
        ApplyTemplate("T1", rational="disconnection"),
        Stop()
    ])

    cfg = ExecutionConfig(template_registry=reg, inventory=stock)

    # 3) 执行程序 -> 顺序化路线
    route = prog.execute(target_smiles=target, config=cfg)
    print(route.to_json())
    print(route.is_solved(stock))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()


#就是：我有一个自定义库存数据库，里面存放的是可购分子；
# 有一个自定义的 retro 模板库，里面存放的是一些 retro 模板；
#我定义了一个 Decision Program，告诉解释器我要怎么做 retrosynthesis。
#     而这个DP也是按照我定义好的模板来执行，执行的操作名称也是自己定义 rational。

#z这个DP与我要实现的研究任务 GP之间的关系：
   #理解DP是基因 的编码方式（基因型）；GP所要考虑的就是如何搜索/进化这些程序（基因）的方法；
     #两者相互配合，GP 在程序空间里进化 DP；DP 被解释器执行后产出一条具体的逆合成路线（表型）。
# DP：问题表示/怎么做逆合成；
# Interpreter(executor)：把DP变成顺序化路线（表型）
# GP : 在DP空间里搜索/进化好的DP
