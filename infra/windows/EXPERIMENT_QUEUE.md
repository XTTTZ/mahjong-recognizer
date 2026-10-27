# 串行实验队列

`experiment_queue.py`只管理本次启动的子进程，用于Windows开机时连续执行多组本地GPU实验。它不登录SSH、不部署云GPU，也不依赖聊天窗口保持打开。应由现有WMI后台启动方式运行；`watch_queue.py`只读状态，关闭监看不会停止队列。

## 配置合同

示例仅展示字段，实际路径、参数与估时由本次实验计划填写：

```json
{
  "schemaVersion": 1,
  "queue_id": "mahjong-v4-experiments",
  "queue_dir": "D:/mahjong/queues/mahjong-v4-experiments",
  "deadline_hours": 20,
  "reserve_minutes": 15,
  "poll_seconds": 2,
  "telemetry_seconds": 60,
  "wait_timeout_minutes": 15,
  "gpu_policy": "exclusive",
  "wait_for_status": ["D:/mahjong/runs/rfdetr-small960-full-v3-20261026/status.json"],
  "env": {"RF_HOME": "D:/mahjong/models", "HF_HUB_OFFLINE": "1"},
  "experiments": [
    {
      "id": "rf-small-640-v4",
      "run_dir": "D:/mahjong/runs/rf-small-640-v4",
      "cwd": "D:/mahjong/code-v4",
      "estimated_minutes": 75,
      "timeout_minutes": 120,
      "steps": [
        {"name": "train", "argv": ["C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe", "-m", "ml.train.rfdetr_train", "--dataset", "D:/mahjong/data/prepared-v4", "--output", "D:/mahjong/runs/rf-small-640-v4"]},
        {"name": "validate", "argv": ["C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe", "-m", "ml.eval.rfdetr_evaluate", "--dataset", "D:/mahjong/data/prepared-v4", "--run", "D:/mahjong/runs/rf-small-640-v4/run.json", "--checkpoint", "D:/mahjong/runs/rf-small-640-v4/checkpoint_best_total.pth", "--out", "D:/mahjong/runs/rf-small-640-v4/validation"]}
      ],
      "report_paths": ["D:/mahjong/runs/rf-small-640-v4/validation/report.json"]
    }
  ]
}
```

`deadline_hours`最大24，从首次启动开始计算，重启队列不会延长截止时间。`estimated_minutes`用于开始新实验前的预算判断；`timeout_minutes`限制整组的训练与评估总时长，step可另设更短的`timeout_minutes`。全局、experiment、step各自可有`env`，按此顺序合并；不要在参数或配置中放凭据。

每组`run_dir`必须独立、首次运行时不存在。队列会创建其中的`status.json`、`stdout.log`、`stderr.log`，训练代码继续保存自己的`run.json`、`metrics.csv`及checkpoint。不要把当前正在跑的v3目录用作新实验目录。

每个step使用argv数组、`shell=False`运行。前一step退出后才开始下一step；某组返回非零、超时或报告缺失会留档，并继续下一组。不是用固定sleep估计前一组是否完成。`report_paths`为精确路径列表；自动汇总仅接受`split=validation_only`报告，同时按验证标注hash与实际图片tree hash分组，避免给不同数据集的分数混排。缺少图片字节来源证据的报告保留记录但不进入模型排名。

## 等待、预算与进程边界

- 默认`gpu_policy=exclusive`：先等待指定旧任务状态结束，再读取`nvidia-smi -q -x`的process类型；明确G作为图形进程记录。Windows驱动617/WDDM实测把桌面应用也标为C+G，故可配置`allowed_gpu_process_paths`，只允许当场核验的完整绝对exe路径，规范化斜杠和大小写后精确匹配；不接受通用进程名称或Python/torchrun等ML运行时。未列路径的C/C+G/unknown仍阻塞。允许的桌面后台逐项记录到等待事件和GPU遥测，不声称GPU物理完全空闲。所有桌面进程都不会被停止。CPU合成测试才使用`disabled`。
- 旧状态文件写着running但child已不存在时，可放行后续GPU空闲检查；文件缺失、活进程或GPU仍忙时最多等待`wait_timeout_minutes`。超时状态为`blocked`，不会擅自抢GPU或停止别人的Python。
- 新实验只有在剩余时间足够其估时加预留时间时才启动；不足则记`not_started_budget`。24小时上限同时受UTC截止和本进程monotonic时钟约束，时钟回拨不能无限延长任务。
- 正在运行的step触及自己的超时或硬截止时，只停止当前`Popen`对应进程树，保留已完成checkpoint。最新checkpoint可能正在写入，恢复前仍需检查完整性。
- 队列期间临时调用Windows防系统休眠，退出时释放；不改电源计划，也不阻止屏幕休眠。

## 重启和失败记录

队列保存配置hash、每个实验hash以及逐组receipt，使用Windows/POSIX系统文件锁防止两个协调器运行同一队列。配置变更需要新队列目录；不能修改旧配置后冒充原计划继续。

相同队列重新启动会跳过已记录的成功、失败、超时或预算跳过实验，不会只因run目录存在就声称完成。上次协调器意外退出但child还活着时，只等它退出，绝不按旧PID发kill（PID可能已复用），也不启动重复训练。残留实验记录为`interrupted`并保留部分产物；是否从完整checkpoint恢复需另行显式安排，不会静默把一次中断当成成功。

`state=completed`表示队列已处理完列表，并不保证每组成功。应同时查看`result`与`experiment_counts`：`all_succeeded`、`completed_with_failures`或`budget_limited`。原始`events.jsonl`记录step起止、GPU采样、失败原因；`summary.json`保留逐组结果、报告hash与同验证标注及图片字节集合内的比较。

## 启动和监看

```text
python -m infra.windows.experiment_queue --config <queue.json>
python -m infra.windows.watch_queue --queue <queue-directory> --follow
```

Windows直接执行对应`.py`路径同样可用。监看显示队列状态、当前模型、step、最近epoch loss、验证mAP、GPU占用及计划剩余时间。ETA是估计，不是保证；未开始实验按计划估时计入，当前RF训练在有epoch记录后按观测速度估算。`--json`用于自动复查，去掉`--follow`只读取一次。

按Ctrl+C只退出**监看**。不要在实际队列协调器终端随意按Ctrl+C；协调器收到中断会停止自己的当前子进程，并记录`interrupted`。断开SSH不影响由WMI启动的后台协调器。

合成测试：`python -m unittest discover -s infra/windows/tests -v`。覆盖真实小CPU子进程的失败后继续、训练后验证、超时且不影响无关进程、重启幂等、活旧PID阻塞、预算拒绝、GPU空闲等待与不同验证集不混排；不把这些测试当作Windows实际GPU队列验收。

## 准备好的多架构目录与显式提交

新版`prepare_experiment_suite.py`生成5/12/24小时预算的独立配置、`check.cmd`、`submit.cmd`和`watch.cmd`，不启动训练。架构、对照因素、参数、估时与一条SSH提交命令见[正式实验目录](../../docs/models/EXPERIMENT_SUITE.md)。`launch_prepared_queue.py`不带`--start`只做CPU完整性和参数预检；只有用户显式执行`--start`才创建WMI后台队列。

job可增加`depends_on:["earlier-experiment-id"]`，只能引用本配置前面出现的唯一ID，禁止前向依赖和环。依赖必须有`succeeded`receipt；缺失、失败、超时、中断或预算跳过会使该job记录`skipped_dependency`，不创建训练目录，并继续处理后面的独立实验。此机制用于两阶段真实检测框评估，共用分类器不会重复训练。依赖跳过属于未成功结果，`state=completed`仍只表示列表已处理。

启动前还校验冻结的全部源文件、数据metadata和官方资产SHA、两个Python环境完整package pins和`pip check`、train/valid像素、衍生YOLO/crop合同。`validate_cli.py`先解析所有训练/评估argv后立即中止在参数边界，再运行显式`--preflight-only`的CPU检查，避免把命令中的参数拼写问题留到无人值守时才发现。历史`specs/`及curated-v3旧`.plan.json`不会加入新离线包；新suite必须在本次代码与环境冻结后单独生成。

## 2026-10-27：v5监看修复

旧监看器在显示第二行epoch时，会因覆盖snapshot变量而抛出`TypeError: string indices must be integers`。这只影响显示，WMI队列及训练进程仍继续。修复已增加多epoch、RF分行CSV和零指标的回归测试，并在Windows真实多轮日志验证。

为保持正在运行的v5冻结源码hash不变，修正版独立部署到`D:/mahjong/tools/watch_queue-20261027.py`；三个已准备suite的`watch.cmd`改指向该只读文件，原命令不变。旧wrapper备份为`.before-watchfix-20261027`。没有修改queue配置或重启模型。重建旧v5环境时，应从仓库复制修正版后再使用这些wrapper：

```sh
scp infra/windows/watch_queue.py win:D:/mahjong/tools/watch_queue-20261027.py
```

先确认Windows的`D:/mahjong/tools`目录存在。未来新生成的bundle则直接包含修正后的`infra/windows/watch_queue.py`。具体部署hash和不间断验收记录见`docs/models/runs/watch-fix-20261027.json`。
