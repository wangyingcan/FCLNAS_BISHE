import copy


class RuntimeContext:
    """运行期上下文，存放恢复计划、进度回调和非配置型状态。"""

    def __init__(self):
        self.resume_task_plan = None
        self.progress_callback = None
        self.auto_resume_plan = None
        self.timing = {}

    def reset_for_task(self):
        self.resume_task_plan = None
        self.progress_callback = None
        self.timing = {}

    def set_resume_task_plan(self, plan):
        self.resume_task_plan = copy.deepcopy(plan) if plan is not None else None

    def clear_resume_task_plan(self):
        self.resume_task_plan = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def clear_progress_callback(self):
        self.progress_callback = None

    def update_timing(self, scope: str, **metrics):
        bucket = self.timing.setdefault(scope, {})
        for key, value in metrics.items():
            bucket[key] = value

