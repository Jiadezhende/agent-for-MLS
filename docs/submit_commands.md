# 提交命令速查

## 配置

```
学号:    23302010024
服务器:  10.176.37.31
```

---

## 完整测试提交流程

### Step 1：申请环境

```powershell
# 启动容器（无 GPU，仅用于上传代码）
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/start" `
  -ContentType "application/json" `
  -Body '{ "id": "23302010024", "gpu": 0 }'
```

### Step 2：获取 SSH 端口

```powershell
# 替换 <mission_id> 为上一步返回的 mission id
Invoke-RestMethod -Method GET `
  -Uri "http://10.176.37.31:8080/submit_status/<mission_id>" `
  | ConvertTo-Json
```

> 从返回的 `info` 字段中找到 `ssh_port=xxxxx`

### Step 3：登录容器，拉取代码

```powershell
ssh root@10.176.37.31 -p <ssh_port> -i $env:USERPROFILE\.ssh\id_ed25519
```

```bash
# 容器内执行
cd /workspace && git pull
# 首次使用则克隆
# git clone https://github.com/Jiadezhende/agent-for-MLS.git /workspace
```

### Step 4：释放环境

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/finish" `
  -ContentType "application/json" `
  -Body '{ "id": "23302010024" }'
```

### Step 5：测试提交

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/submit-test" `
  -ContentType "application/json" `
  -Body '{ "id": "23302010024", "gpu": 1 }'
```

> 保存返回的 `output_file`

### Step 6：查看提交状态

```powershell
Invoke-RestMethod -Method GET `
  -Uri "http://10.176.37.31:8080/submit_status/<output_file>" `
  | ConvertTo-Json
```

状态值：`running` / `succeeded` / `failed` / `killed`

### Step 7：查看输出文件

浏览器打开：[http://10.176.37.31:8080/outputs](http://10.176.37.31:8080/outputs)

---

## 正式提交（最多 2 次，GPT-5.4）

> 截止：4/28 8am 前 2 次，**4/21 8am 前至少 1 次**

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/submit" `
  -ContentType "application/json" `
  -Body '{ "id": "23302010024", "gpu": 1 }'
```

---

## 其他常用命令

```powershell
# 检查 GPU 资源
Invoke-RestMethod -Method GET -Uri "http://10.176.37.31:8080/list"
```
