# utils.py
import logging
import inspect
import os
from functools import wraps
from datetime import datetime

# 用于生成独特的日志文件名
def get_log_filename():
    log_dir = "logs_wyc"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 使用时间戳来命名日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"日志_{timestamp}.txt")

    return log_filename

# 配置日志记录
def setup_logger():
    log_filename = get_log_filename()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return log_filename

# 装饰器：记录函数调用和执行过程
def log_function_call(func):
    """
    装饰器：在函数执行前后记录日志，包括函数名、类名、参数、执行结果、代码位置。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取函数名称
        func_name = func.__name__

        # 获取类名：通过第一个参数（self）来获得类名
        class_name = args[0].__class__.__name__ if args else "未知类"

        # 获取调用者信息
        frame = inspect.currentframe().f_back
        caller = inspect.getframeinfo(frame).function
        
        # 获取方法的文件和行号
        file_name = inspect.getfile(func)  # 文件路径
        line_number = inspect.getsourcelines(func)[1]  # 方法定义的行号

        # 创建可点击的文件路径链接，支持 vscode 跳转
        # file_url = f"file://{os.path.abspath(file_name)}:{line_number}"
        
        logging.info(f"调用函数：{func_name} 来自类：{class_name}")
        
        try:
            # 执行函数本身
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # 如果发生异常，记录错误日志
            logging.info(f"调用函数：{func_name} 来自类：{class_name} 中执行失败，错误信息：{str(e)}")
            # raise e 

    return wrapper
