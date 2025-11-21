# run_feas_demo.py
import sys
import traceback

from gp_retro_repr import (
    Program, Select, ApplyTemplate, Stop
)
from gp_retro_feas import (
    TemplateTyping, TemplatePreselector, ActionMaskBuilder, FeasibleExecutor
)

from demo_utils import build_world_t1  # 统一的库存 + 模板 + 目标

def main():
    # 1) 库存与模板（示例）——改为从 demo_utils 统一获取
    stock, reg, target = build_world_t1()

    # 2) 家族需求（可选但推荐）
    typing = TemplateTyping()
    typing.set_template_family("T1", "oxidation")
    typing.set_family_requirement("oxidation", {"alcohol"})

    # 3) 预筛 + 掩码
    pre = TemplatePreselector(reg, typing=typing).preselect(target)
    print("FG gating:", pre.reasoning)
    mask = ActionMaskBuilder(reg, inventory=stock).build(target)
    print("Feasible templates:", mask.feasible_templates)

    # 4) 可行执行（失败自动修复）
    exe = FeasibleExecutor(reg, inventory=stock)
    prog = Program([Select(0), ApplyTemplate("T1", rational="demo"), Stop()])
    route = exe.execute(prog, target_smiles=target)
    print(route.to_json())
    print("Solved:", route.is_solved(stock))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 确保任何错误都能在控制台看到
        traceback.print_exc()
        sys.exit(1)


#库存+反应模板；
#家族需求：给模板打上标签，并定义每个标签对应的官能团需求；
    #作用---在真正调用 rdchiral之前,先根据目标分子的官能团信息筛选出可能用到的模板，
    # 减少不必要的 rdchiral 调用，提高效率；(先一步过滤掉不可能的模板)

#预筛选+掩码（候选动作集合）：
   #emplatePreselector：完成 用功能团+家族需求 做快速筛选(不调用 rdchiral)； 
   # ActionMaskBuilder：在预筛结果上，逐个模板对 CCO 做 rdchiral 适用性检查，可选再加库存门控/反应物数阈值，得到可行模板掩码。
         # 作用
            # 给 GP 的变异/交叉提供“只能从这些可行模板里挑”的候选池，避免不可行的无效操作，极大压缩搜索空间。
        # # 对应输出
             # 第二行 Feasible templates: ['T1']：说明对 CCO 来说只有 T1 是可行动作。

#执行调用:
     # 用 FeasibleExecutor 执行 DP 程序：
               # Select(0) 选中当前集合里的第 0 个分子（起始就是目标 CCO）；
     # ApplyTemplate("T1") 对它应用 T1；
# 若模板失败且开启 repair_on_failure=True，会自动从可行掩码里换一个可行模板（这里第一次就成功，所以没修复）。
   # 生成顺序化路线的一条记录。
# 作用
          # 把“按构造可行”落地到执行层；评估时也能更稳健（少量坏基因可被修复）。
          # 产出你要的路线 JSON，可直接进入评分/奖励/日志链路。
