# SSH密钥配置说明

## ✅ 已生成SSH密钥

SSH密钥对已在本地生成，位置：
- 私钥: `~/.ssh/id_ed25519`
- 公钥: `~/.ssh/id_ed25519.pub`
- 指纹: `SHA256:JY6UZRu86xyARDB2kgt0FPyKgb+k994S2AKZeFRpDGo`

## 📋 需要添加到GitHub的公钥

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKY89relJTLt72Tzq1qeiPAEtqQrrkek0hvi7ITDy+2b xuenailao@github
```

## 🔧 添加步骤

### 1. 访问GitHub SSH设置页面
https://github.com/settings/keys

### 2. 点击 "New SSH key"

### 3. 填写信息
- **Title**: `AAD-Development-Key` (或任意描述性名称)
- **Key type**: Authentication Key
- **Key**: 复制上面的整行公钥（从 `ssh-ed25519` 开始到 `xuenailao@github` 结束）

### 4. 点击 "Add SSH key"

### 5. 确认添加成功
您会在SSH keys列表中看到新添加的密钥

## ✅ 验证连接

添加密钥后，在终端运行：

```bash
# 测试SSH连接
ssh -T git@github.com
```

成功的话会看到：
```
Hi xuenailao! You've successfully authenticated, but GitHub does not provide shell access.
```

## 🚀 推送到GitHub

SSH配置完成后：

```bash
cd /home/junruw2/AAD

# 切换回SSH URL
git remote set-url origin git@github.com:xuenailao/JPM-Practicum-AAD.git

# 推送
git push -u origin main
```

## 📝 问题排查

### 如果看到 "Permission denied (publickey)"
1. 确认公钥已添加到GitHub
2. 检查密钥权限: `ls -la ~/.ssh/`
   - 私钥应该是 `-rw-------` (600)
   - 公钥应该是 `-rw-r--r--` (644)
3. 重新测试: `ssh -T git@github.com`

### 如果看到 "Host key verification failed"
```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

### 查看详细连接日志
```bash
ssh -vT git@github.com
```

## 📌 重要说明

1. **不要分享私钥**: `id_ed25519` 是私钥，永远不要分享或上传
2. **只分享公钥**: `id_ed25519.pub` 是公钥，可以安全地添加到GitHub
3. **密钥已生成**: 本地环境已经有完整的密钥对

## 🔄 当前Git状态

```
Repository: /home/junruw2/AAD
Branch: main
Commit: ccfe139
Remote: https://github.com/xuenailao/JPM-Practicum-AAD.git (当前)
准备切换到: git@github.com:xuenailao/JPM-Practicum-AAD.git (SSH)
```

---

**下一步**: 将上面的公钥添加到GitHub → 测试连接 → 推送代码
