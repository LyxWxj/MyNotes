---
type: Note
related_to: "[[github-cli]]"
status: Active
---

# GitHub CLI (gh) 常见操作速查

## 安装与认证

```bash
# 安装（Ubuntu/Debian）
sudo apt install gh

# 安装（macOS）
brew install gh

# 认证登录
gh auth login                   # 交互式登录
gh auth login --with-token      # 通过 token 登录（CI 场景）
gh auth status                  # 查看认证状态
gh auth refresh                 # 刷新认证凭据
gh auth switch                  # 切换账号
gh auth logout                  # 登出
```

---

## 一、仓库操作

### 创建仓库

```bash
gh repo create                          # 交互式创建
gh repo create my-project               # 创建公开仓库
gh repo create my-project --private     # 创建私有仓库
gh repo create my-project --public --description "描述"
gh repo create my-project --clone       # 创建并克隆到本地
gh repo create my-project --template owner/template-repo  # 从模板创建
```

### 查看仓库

```bash
gh repo view                            # 查看当前仓库信息
gh repo view owner/repo                 # 查看指定仓库
gh repo view --web                      # 在浏览器中打开仓库
gh repo list                            # 列出自己的仓库
gh repo list owner                      # 列出指定用户的仓库
gh repo list --limit 50                 # 限制数量
```

### 克隆与 Fork

```bash
gh repo clone owner/repo                # 克隆仓库
gh repo clone owner/repo ~/path         # 克隆到指定路径
gh repo fork owner/repo                 # Fork 仓库
gh repo fork owner/repo --clone         # Fork 并克隆
gh repo fork owner/repo --remote-name upstream  # 指定远程名
```

### 其他仓库操作

```bash
gh repo archive owner/repo              # 归档仓库
gh repo delete owner/repo               # 删除仓库（危险）
gh repo sync                            # 同步 fork 与上游
gh repo rename new-name                 # 重命名当前仓库
gh repo edit --description "新描述"     # 修改仓库描述
gh repo edit --visibility private       # 修改可见性
```

---

## 二、Issue 操作

### 创建 Issue

```bash
gh issue create                         # 交互式创建
gh issue create --title "Bug" --body "描述"
gh issue create --label "bug,priority:high"
gh issue create --assignee "@me"
gh issue create --project "My Board"
gh issue create --template bug_report   # 使用 issue 模板
```

### 查看 Issue

```bash
gh issue list                            # 列出 issues
gh issue list --state open               # 只看打开的
gh issue list --state closed             # 只看关闭的
gh issue list --label "bug"              # 按标签筛选
gh issue list --assignee "@me"           # 指定负责人
gh issue list --author "@me"             # 指定创建者
gh issue view 123                        # 查看 issue 详情
gh issue view 123 --comments             # 包含评论
```

### 管理 Issue

```bash
gh issue close 123                       # 关闭 issue
gh issue close 123 --reason "completed"  # 带原因关闭
gh issue reopen 123                      # 重新打开
gh issue edit 123 --add-label "verified" # 添加标签
gh issue edit 123 --remove-label "wip"   # 移除标签
gh issue edit 123 --add-assignee "user"  # 添加负责人
gh issue edit 123 --milestone "v1.0"     # 设置里程碑
gh issue delete 123                      # 删除 issue
```

### Issue 评论

```bash
gh issue comment 123 --body "评论内容"
gh issue comment 123 --body-file comment.md  # 从文件读取内容
```

---

## 三、Pull Request 操作

### 创建 PR

```bash
gh pr create                              # 交互式创建
gh pr create --title "feat: 新功能" --body "详细描述"
gh pr create --base main --head feature    # 指定目标和源分支
gh pr create --draft                       # 创建草稿 PR
gh pr create --reviewer "user1,user2"      # 指定 reviewer
gh pr create --assignee "@me"
gh pr create --label "feature,needs-review"
gh pr create --milestone "v1.0"
gh pr create --no-editor                   # 跳过编辑器，直接提交
gh pr create --fill                        # 自动从提交信息填充标题和描述
gh pr create --fill-first                  # 用第一个提交填充
```

### 查看 PR

```bash
gh pr list                                # 列出 PRs
gh pr list --state open
gh pr list --state merged
gh pr list --state closed
gh pr list --author "@me"
gh pr list --reviewer "@me"
gh pr list --base main
gh pr list --label "needs-review"
gh pr view 123                            # 查看 PR 详情
gh pr view 123 --comments                 # 包含评论
gh pr view 123 --web                      # 在浏览器中打开
gh pr view 123 --json title,body,state    # JSON 格式输出
```

### PR 审查

```bash
gh pr review 123 --approve                # 批准
gh pr review 123 --request-changes --body "需要修改的地方"
gh pr review 123 --comment --body "一般评论"
gh pr review 123 --approve -b "LGTM!"     # 批准并评论
```

### PR 合并与管理

```bash
gh pr merge 123                           # 合并（自动选择策略）
gh pr merge 123 --merge                   # 创建合并提交
gh pr merge 123 --squash                  # 压缩合并
gh pr merge 123 --rebase                  # 变基合并
gh pr merge 123 --auto                    # 设置自动合并（CI 通过后）
gh pr merge 123 --delete-branch           # 合并后删除分支
gh pr close 123                           # 关闭 PR
gh pr reopen 123                          # 重新打开 PR
```

### PR 状态检查

```bash
gh pr checks 123                          # 查看 CI 状态
gh pr checks 123 --watch                  # 持续监控直到完成
gh pr diff 123                            # 查看 PR diff
```

### 检出 PR

```bash
gh pr checkout 123                        # 检出 PR 到本地
gh pr checkout 123 --branch my-branch     # 指定本地分支名
```

---

## 四、GitHub Actions

### 查看工作流

```bash
gh workflow list                          # 列出所有工作流
gh workflow view <workflow>               # 查看工作流详情
gh workflow view <workflow> --yaml        # 查看 YAML 定义
```

### 运行工作流

```bash
gh workflow run <workflow>                # 触发工作流
gh workflow run <workflow> --ref main     # 指定分支
gh workflow run <workflow> -f key=value   # 传入输入参数
gh workflow run deploy -f environment=staging -f version=1.2.3
```

### 查看运行记录

```bash
gh run list                               # 列出最近的运行
gh run list --workflow=<workflow>         # 按工作流筛选
gh run list --status failure              # 按状态筛选
gh run list --limit 10
gh run view <run-id>                      # 查看运行详情
gh run view <run-id> --log                # 查看日志
gh run view <run-id> --job <job-id>       # 查看特定 job
gh run view <run-id> --web                # 在浏览器中打开
```

### 管理运行

```bash
gh run watch <run-id>                     # 实时监控运行状态
gh run rerun <run-id>                     # 重新运行
gh run rerun <run-id> --failed            # 只重跑失败的 job
gh run cancel <run-id>                    # 取消运行
gh run download <run-id>                  # 下载产物
gh run download <run-id> -n artifact-name # 指定产物名
```

### 查看缓存

```bash
gh cache list                             # 列出缓存
gh cache list --limit 50
gh cache delete <cache-id>                # 删除缓存
gh cache delete --all                     # 清空所有缓存
```

---

## 五、Gist 操作

```bash
gh gist list                              # 列出 gists
gh gist create <file>                     # 创建 gist
gh gist create <file> -d "描述"           # 带描述
gh gist create <file> --public            # 公开 gist
gh gist create file1.md file2.py          # 多文件 gist
gh gist view <id>                         # 查看 gist
gh gist edit <id>                         # 编辑 gist
gh gist clone <id>                        # 克隆 gist
gh gist delete <id>                       # 删除 gist
```

---

## 六、GitHub Releases

```bash
gh release list                           # 列出 releases
gh release view <tag>                     # 查看 release 详情
gh release create v1.0.0                  # 创建 release
gh release create v1.0.0 --title "v1.0.0" --notes "发布说明"
gh release create v1.0.0 --notes-file CHANGELOG.md
gh release create v1.0.0 --draft          # 创建草稿 release
gh release create v1.0.0 --prerelease     # 标记为预发布
gh release create v1.0.0 ./dist/*         # 上传产物文件
gh release edit v1.0.0 --notes "更新说明" # 修改 release
gh release delete v1.0.0                  # 删除 release
gh release download v1.0.0                # 下载 release 产物
gh release download v1.0.0 -p "*.tar.gz"  # 模式匹配下载
```

---

## 七、搜索

```bash
# 搜索仓库
gh search repos "machine learning" --language=python --stars=">100"

# 搜索代码
gh search code "function_name" --repo owner/repo

# 搜索 issues
gh search issues "bug crash" --state=open --label=priority:high

# 搜索 PRs
gh search prs "feature" --state=open --review-requested="@me"

# 搜索 commits
gh search commits "fix" --author="user"
```

---

## 八、SSH Key 管理

```bash
gh ssh-key list                           # 列出 SSH keys
gh ssh-key add ~/.ssh/id_ed25519.pub      # 添加 SSH key
gh ssh-key add ~/.ssh/id_ed25519.pub -t "My Key"  # 带标题
gh ssh-key delete <key-id>               # 删除 SSH key
```

---

## 九、GPG Key 管理

```bash
gh gpg-key list                           # 列出 GPG keys
gh gpg-key add <key-file>                 # 添加 GPG key
gh gpg-key delete <key-id>               # 删除 GPG key
```

---

## 十、API 与扩展

### 直接调用 GitHub API

```bash
gh api repos/owner/repo                   # GET 请求
gh api repos/owner/repo/issues -f state=open  # 带参数
gh api graphql -f query='{ viewer { login } }'  # GraphQL 查询
gh api repos/owner/repo -X POST -f name="new-repo"  # POST 请求
gh api repos/owner/repo -X PATCH -f description="updated"  # PATCH
gh api repos/owner/repo/releases/latest   # 获取最新 release
gh api user/repos --paginate              # 分页获取所有仓库
gh api repos/owner/repo/issues --jq '.[].title'  # 用 jq 过滤输出
```

**使用场合**: `gh api` 是万能命令，任何 GitHub REST/GraphQL API 都可以通过它调用，适合自动化脚本。

### 扩展管理

```bash
gh extension list                         # 列出已安装扩展
gh extension install owner/gh-extension   # 安装扩展
gh extension upgrade                      # 升级所有扩展
gh extension remove <extension>           # 删除扩展
gh extension browse                       # 浏览可用扩展
```

常用扩展推荐：
- `gh dash` — 终端仪表盘
- `gh copilot` — AI 辅助命令
- `gh act` — 本地运行 Actions
- `gh-markdown-preview` — 预览 Markdown

---

## 十一、实用别名与配置

```bash
# 设置别名
gh alias set pv 'pr view'
gh alias set prs 'pr list --state open --reviewer @me'
gh alias set myissues 'issue list --assignee @me --state open'
gh alias set mrs 'run list --limit 5'

# 配置默认编辑器
gh config set editor vim
gh config set editor "code --wait"

# 配置默认浏览器
gh config set browser firefox

# 配置默认协议
gh config set git_protocol ssh

# 查看所有配置
gh config list
```

---

## 十二、常见工作流示例

### 完整的 PR 工作流

```bash
# 1. 创建功能分支
git switch -c feat/new-feature

# 2. 开发并提交
git add .
git commit -m "feat: add new feature"

# 3. 推送分支
git push -u origin feat/new-feature

# 4. 创建 PR
gh pr create --fill --reviewer teammate

# 5. 等待 CI 通过并监控
gh pr checks --watch

# 6. 查看审查意见
gh pr view --comments

# 7. 合并
gh pr merge --squash --delete-branch
```

### 热修复工作流

```bash
# 1. 从 main 创建修复分支
git switch -c hotfix/critical-bug main

# 2. 修复并提交
git add .
git commit -m "fix: critical bug in auth"

# 3. 推送并创建 PR
git push -u origin hotfix/critical-bug
gh pr create --base main --title "hotfix: critical bug" --reviewer @me

# 4. 快速审查并合并
gh pr review --approve
gh pr merge --squash

# 5. 创建 hotfix release
gh release create v1.0.1 --title "Hotfix v1.0.1" --notes "修复认证关键 bug"
```

### Issue 驱动开发

```bash
# 1. 查看分配的 issue
gh issue list --assignee "@me"

# 2. 创建分支并关联 issue
git switch -c fix/issue-123
git commit -m "fix: resolve login timeout

Closes #123"

# 3. 推送并创建 PR（自动关联 issue）
git push -u origin fix/issue-123
gh pr create --fill

# 4. 合并后 issue 自动关闭
gh pr merge --squash
```

### 本地调试 Actions

```bash
# 使用 gh act 扩展（需安装 nektos/act）
gh extension install nektos/act

# 运行默认 workflow
gh act

# 运行特定 workflow
gh act push -W .github/workflows/ci.yml

# 运行特定 job
gh act -j test
```
