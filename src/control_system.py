"""
远程声音控制系统 - 控制系统模块（PID控制）
Control System Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """系统状态"""
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    time: float = 0.0


class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.5, kd: float = 0.2,
                 setpoint: float = 0.0, dt: float = 0.01):
        """
        初始化PID控制器
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            setpoint: 目标值
            dt: 采样时间
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_output = 0.0
        self.output_history = []
        self.error_history = []
    
    def update(self, current_value: float) -> float:
        """
        计算PID输出
        
        Args:
            current_value: 当前系统值
            
        Returns:
            output: 控制器输出
        """
        # 计算误差
        error = self.setpoint - current_value
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项
        self.integral_error += error * self.dt
        i_term = self.ki * self.integral_error
        
        # 微分项
        d_term = self.kd * (error - self.prev_error) / (self.dt + 1e-10)
        
        # 总输出
        output = p_term + i_term + d_term
        
        # 记录历史
        self.prev_error = error
        self.prev_output = output
        self.output_history.append(output)
        self.error_history.append(error)
        
        return output
    
    def set_setpoint(self, setpoint: float) -> None:
        """设置目标值"""
        self.setpoint = setpoint
    
    def reset(self) -> None:
        """重置控制器"""
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.output_history = []
        self.error_history = []
    
    def get_stats(self) -> Dict:
        """获取控制统计信息"""
        if len(self.error_history) == 0:
            return {}
        
        errors = np.array(self.error_history)
        outputs = np.array(self.output_history)
        
        return {
            'mean_error': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'steady_state_error': np.abs(errors[-1]) if len(errors) > 0 else 0,
            'mean_output': np.mean(outputs),
            'max_output': np.max(outputs),
            'min_output': np.min(outputs),
        }


class DCMotor:
    """直流电动机模型"""
    
    def __init__(self, inertia: float = 0.1, friction: float = 0.5, 
                 torque_constant: float = 1.0):
        """
        初始化电动机模型
        
        Args:
            inertia: 转动惯量
            friction: 摩擦系数
            torque_constant: 力矩常数
        """
        self.inertia = inertia
        self.friction = friction
        self.torque_constant = torque_constant
        
        self.state = SystemState()
        self.max_voltage = 10.0
        self.max_velocity = 10.0
    
    def apply_voltage(self, voltage: float, dt: float = 0.01) -> SystemState:
        """
        对电动机施加电压
        
        Args:
            voltage: 施加的电压
            dt: 时间步长
            
        Returns:
            state: 更新后的系统状态
        """
        # 限制电压
        voltage = np.clip(voltage, -self.max_voltage, self.max_voltage)
        
        # 生成力矩
        torque = self.torque_constant * voltage
        
        # 摩擦力矩
        friction_torque = -self.friction * self.state.velocity
        
        # 总力矩
        total_torque = torque + friction_torque
        
        # 角加速度 (ω = τ/J)
        self.state.acceleration = total_torque / (self.inertia + 1e-10)
        
        # 更新角速度
        self.state.velocity += self.state.acceleration * dt
        self.state.velocity = np.clip(self.state.velocity, 
                                     -self.max_velocity, self.max_velocity)
        
        # 更新位置
        self.state.position += self.state.velocity * dt
        self.state.time += dt
        
        return self.state
    
    def reset(self) -> None:
        """重置电动机状态"""
        self.state = SystemState()
    
    def get_state(self) -> SystemState:
        """获取当前状态"""
        return self.state


class ControlSystem:
    """完整的控制系统"""
    
    def __init__(self, system_type: str = 'motor'):
        """
        初始化控制系统
        
        Args:
            system_type: 系统类型（'motor', 'robotic_arm' 等）
        """
        self.system_type = system_type
        
        # PID控制器
        self.controller = PIDController(kp=2.0, ki=0.5, kd=0.3)
        
        # 系统模型
        if system_type == 'motor':
            self.system = DCMotor(inertia=0.1, friction=0.5)
        else:
            self.system = DCMotor()
        
        # 历史记录
        self.position_history = []
        self.velocity_history = []
        self.control_history = []
        self.time_history = []
    
    def execute_command(self, command: str, duration: float = 1.0, dt: float = 0.01) -> bool:
        """
        执行控制命令
        
        Args:
            command: 命令字符串
            duration: 执行时长（秒）
            dt: 采样时间
            
        Returns:
            success: 是否成功执行
        """
        # 解析命令
        setpoint = self._parse_command(command)
        if setpoint is None:
            logger.error(f"Unknown command: {command}")
            return False
        
        # 重置控制器
        self.controller.reset()
        self.system.reset()
        self.position_history = []
        self.velocity_history = []
        self.control_history = []
        self.time_history = []
        
        # 设置目标
        self.controller.set_setpoint(setpoint)
        
        # 执行控制循环
        num_steps = int(duration / dt)
        for step in range(num_steps):
            # 获取当前位置
            current_position = self.system.get_state().position
            
            # PID控制
            control_output = self.controller.update(current_position)
            
            # 应用到系统
            self.system.apply_voltage(control_output, dt)
            
            # 记录历史
            state = self.system.get_state()
            self.position_history.append(state.position)
            self.velocity_history.append(state.velocity)
            self.control_history.append(control_output)
            self.time_history.append(state.time)
        
        logger.info(f"Command '{command}' executed successfully")
        logger.info(f"Final position: {self.position_history[-1]:.4f}")
        
        return True
    
    def _parse_command(self, command: str) -> float:
        """
        解析控制命令
        
        Args:
            command: 命令字符串
            
        Returns:
            setpoint: 目标值，如果命令无效则返回None
        """
        command = command.lower().strip()
        
        # 预定义的命令映射
        command_map = {
            'forward': 5.0,
            'backward': -5.0,
            'stop': 0.0,
            'rotate': 3.0,
            'rotate_left': 3.0,
            'rotate_right': -3.0,
            'left': 3.0,
            'right': -3.0,
            'up': 5.0,
            'down': -5.0,
            'speed_up': 8.0,
            'speed_down': 2.0,
            'full_speed': 10.0,
            'half_speed': 5.0,
            'quarter_speed': 2.5,
        }
        
        # 检查预定义命令
        if command in command_map:
            return command_map[command]
        
        # 尝试解析数值命令（例如 "setpoint:7.5"）
        if ':' in command:
            try:
                parts = command.split(':')
                if len(parts) == 2 and parts[0] == 'setpoint':
                    return float(parts[1])
            except (ValueError, IndexError):
                pass
        
        return None
    
    def get_performance_metrics(self) -> Dict:
        """获取系统性能指标"""
        if len(self.position_history) == 0:
            return {}
        
        positions = np.array(self.position_history)
        setpoint = self.controller.setpoint
        
        # 稳态误差
        steady_state_error = abs(positions[-1] - setpoint)
        
        # 平均误差
        errors = abs(positions - setpoint)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # 超调量
        if positions[0] < setpoint:
            overshoot = max(0, np.max(positions) - setpoint) / (setpoint + 1e-10)
        else:
            overshoot = max(0, positions[0] - np.min(positions)) / (abs(setpoint) + 1e-10)
        
        # 上升时间（从10%到90%）
        min_pos = np.min(positions)
        max_pos = np.max(positions)
        pos_range = max_pos - min_pos
        lower_10 = min_pos + 0.1 * pos_range
        upper_90 = min_pos + 0.9 * pos_range
        
        rise_time = None
        lower_idx = None
        upper_idx = None
        for i, pos in enumerate(positions):
            if lower_idx is None and pos >= lower_10:
                lower_idx = i
            if pos >= upper_90:
                upper_idx = i
                break
        
        if lower_idx is not None and upper_idx is not None:
            rise_time = (upper_idx - lower_idx) * (self.time_history[1] - self.time_history[0] 
                                                    if len(self.time_history) > 1 else 0.01)
        
        metrics = {
            'steady_state_error': steady_state_error,
            'mean_error': mean_error,
            'max_error': max_error,
            'overshoot': overshoot,
            'rise_time': rise_time,
            'final_position': positions[-1],
            'setpoint': setpoint,
        }
        
        return metrics
    
    def print_status(self) -> None:
        """打印系统状态"""
        if len(self.position_history) == 0:
            logger.info("System not yet executed")
            return
        
        metrics = self.get_performance_metrics()
        logger.info("System Performance:")
        for key, value in metrics.items():
            if value is not None:
                logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    # 测试控制系统
    control_sys = ControlSystem(system_type='motor')
    
    # 执行命令
    commands = ['forward', 'stop', 'backward']
    
    for cmd in commands:
        control_sys.execute_command(cmd, duration=2.0, dt=0.01)
        control_sys.print_status()
        print()
