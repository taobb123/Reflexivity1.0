# 快速测试指南

## 方式1：直接运行完整分析（推荐）

如果你已经在当前PowerShell会话中设置了Token，可以直接运行：

```powershell
python analyze_stock.py --stock 平安银行 --weeks 120
```

## 方式2：在当前会话设置Token后测试

```powershell
# 设置Token（仅在当前会话有效）
$env:TUSHARE_TOKEN="your_token_here"

# 运行测试
python test_tushare_interactive.py

# 或直接运行分析
python analyze_stock.py --stock 平安银行 --weeks 120
```

## 方式3：永久设置环境变量（Windows）

### 方法A：PowerShell（当前用户）
```powershell
[System.Environment]::SetEnvironmentVariable('TUSHARE_TOKEN', 'your_token_here', 'User')
```

### 方法B：系统环境变量（需要管理员权限）
1. 右键"此电脑" → "属性"
2. 高级系统设置 → 环境变量
3. 新建系统变量：变量名 `TUSHARE_TOKEN`，变量值 `your_token_here`

设置后需要**重启PowerShell**才能生效。

## 方式4：使用命令行参数传入Token

```powershell
python analyze_stock.py --stock 平安银行 --weeks 120 --token "your_token_here"
```

## 检查Token是否设置成功

```powershell
# 检查环境变量
echo $env:TUSHARE_TOKEN

# 如果有输出（显示Token），说明设置成功
# 如果没有输出，需要重新设置
```

## 常见问题

### Q: Token在哪里获取？
A: 登录 https://tushare.pro/user/index → 接口页面 → 复制Token

### Q: 需要多少积分？
A: 基础功能需要≥120积分。可以通过：
- 实名认证
- 邀请好友
- 充值购买

### Q: 提示"积分不足"怎么办？
A: 检查账户积分，升级账户或充值

### Q: 提示"Token无效"怎么办？
A: 检查Token是否正确复制，是否有空格或换行

