---
type: Note
related_to: "[[git]]"
status: Active
---

# 常见 Git 操作速查

## 一、仓库初始化与克隆

### 初始化新仓库

```bash
git init
```

**使用场合**: 本地新建项目，需要纳入版本控制时。

### 克隆远程仓库

```bash
git clone <repo-url>
git clone <repo-url> <dir-name>       # 指定本地目录名
git clone --depth 1 <repo-url>         # 浅克隆，只取最新提交（适合大型仓库）
```

**使用场合**: 参与已有项目，将远程仓库拉取到本地。

---

## 二、分支管理

### 查看分支

```bash
git branch                    # 查看本地分支
git branch -r                 # 查看远程分支
git branch -a                 # 查看所有分支
git branch -v                 # 查看分支及最后一次提交
```

### 创建与切换

```bash
git branch <name>             # 创建新分支
git switch <name>             # 切换到指定分支（推荐）
git switch -c <name>          # 创建并切换（推荐）
git checkout <name>           # 切换分支（旧语法）
git checkout -b <name>        # 创建并切换（旧语法）
git switch -c <name> origin/<name>  # 基于远程分支创建本地分支
```

**使用场合**: 开发新功能、修复 bug 时，从主分支创建独立分支进行开发。

### 删除分支

```bash
git branch -d <name>          # 删除已合并的本地分支
git branch -D <name>          # 强制删除本地分支
git push origin -d <name>     # 删除远程分支
```

### 重命名分支

```bash
git branch -m <old> <new>     # 重命名本地分支
```

---

## 三、暂存与提交

### 查看状态

```bash
git status                    # 查看工作区状态
git status -s                 # 简洁格式
git diff                      # 查看未暂存的更改
git diff --staged             # 查看已暂存的更改
git diff <branch1> <branch2>  # 比较两个分支的差异
```

### 暂存文件

```bash
git add <file>                # 暂存指定文件
git add .                     # 暂存当前目录所有更改
git add -p                    # 交互式暂存（逐块选择）
git add -u                    # 暂存所有已跟踪文件的修改/删除
```

**使用场合**: `add -p` 适合将一个文件中的多处修改拆分为不同提交，保持提交粒度清晰。

### 提交

```bash
git commit -m "message"           # 提交暂存区内容
git commit -am "message"          # 暂存并提交已跟踪文件（跳过新文件）
git commit --amend                # 修改最近一次提交（消息或内容）
git commit --amend --no-edit      # 追加内容到最近一次提交，不改消息
git commit --allow-empty -m "msg" # 空提交（触发 CI 等场景）
```

### 提交规范（Conventional Commits）

```
<type>(<scope>): <description>

feat: 添加用户登录功能
fix(auth): 修复 token 过期未刷新问题
docs: 更新 README
refactor: 重构数据处理模块
chore: 升级依赖版本
```

---

## 四、远程操作

### 远程仓库配置

```bash
git remote -v                         # 查看远程仓库
git remote add origin <url>           # 添加远程仓库
git remote set-url origin <new-url>   # 修改远程 URL
git remote remove <name>              # 删除远程仓库
```

### 推送与拉取

```bash
git push origin <branch>              # 推送到远程分支
git push -u origin <branch>           # 推送并设置上游跟踪
git push --force-with-lease           # 安全强制推送（检测远程是否有新提交）
git push --force                      # 强制推送（危险，慎用）
git push origin --tags                # 推送标签

git pull                              # 拉取并合并（= fetch + merge）
git pull --rebase                     # 拉取并变基（保持线性历史）
git fetch                             # 只拉取远程数据，不合并
git fetch --prune                     # 拉取并清理已删除的远程分支引用
```

**使用场合**:
- `pull --rebase`: 本地提交尚未共享时，用变基保持提交历史整洁。
- `--force-with-lease`: 变基后推送，比 `--force` 安全，防止覆盖他人提交。

---

## 五、合并与变基

### 合并（Merge）

```bash
git merge <branch>                    # 合并指定分支到当前分支
git merge --no-ff <branch>            # 禁止快进合并，保留分支拓扑
git merge --squash <branch>           # 压缩合并（所有提交合为一个）
git merge --abort                     # 中止合并，恢复到合并前状态
```

**使用场合**: 团队协作中合并功能分支到主分支，`--no-ff` 能清晰保留分支历史。

### 变基（Rebase）

```bash
git rebase <branch>                   # 将当前分支变基到指定分支
git rebase -i HEAD~N                  # 交互式变基最近 N 个提交
git rebase --onto <base> <from> <to>  # 精确控制变基范围
git rebase --abort                    # 中止变基
git rebase --continue                 # 解决冲突后继续变基
```

**使用场合**: 功能分支开发完成后，变基到最新主分支再合并，保持线性历史。

### 交互式变基操作

```
pick   = 保留提交
reword = 修改提交消息
edit   = 暂停提交，允许修改内容
squash = 与前一个提交合并，保留消息
fixup  = 与前一个提交合并，丢弃消息
drop   = 删除提交
```

**使用场合**: 整理本地提交历史，在推送前压缩/重排/修正提交。

---

## 六、撤销与回退

### 工作区撤销

```bash
git restore <file>                    # 撤销工作区文件的修改
git restore .                         # 撤销所有工作区修改
```

### 暂存区撤销

```bash
git restore --staged <file>           # 取消暂存（保留工作区修改）
git restore --staged .                # 取消所有暂存
```

### 提交回退

```bash
git reset --soft HEAD~1               # 撤销提交，保留暂存区和工作区
git reset --mixed HEAD~1              # 撤销提交和暂存，保留工作区（默认）
git reset --hard HEAD~1               # 完全回退（危险，丢失所有更改）
git revert <commit>                   # 创建新提交来撤销指定提交（安全）
git revert <commit1>..<commit2>       # 撤销一个范围的提交
```

**使用场合**:
- `reset --soft`: 提交后发现需要修改，回退但保留已暂存内容。
- `revert`: 已推送到远程的提交，不能用 reset，用 revert 创建反向提交。

### 恢复丢失的提交

```bash
git reflog                            # 查看所有操作历史（包括已回退的提交）
git cherry-pick <commit>              # 拣选特定提交
```

---

## 七、标签管理

```bash
git tag                               # 列出标签
git tag <name>                        # 创建轻量标签
git tag -a <name> -m "message"        # 创建附注标签
git tag -a <name> <commit>            # 为历史提交打标签
git push origin <tag>                 # 推送指定标签
git push origin --tags                # 推送所有标签
git tag -d <name>                     # 删除本地标签
git push origin -d <tag>              # 删除远程标签
```

**使用场合**: 版本发布时，为特定提交打标签标记版本号（如 `v1.0.0`）。

---

## 八、暂存工作区（Stash）

```bash
git stash                             # 暂存当前工作区修改
git stash push -m "描述"              # 带描述信息暂存
git stash list                        # 查看暂存列表
git stash pop                         # 恢复最近一次暂存并删除记录
git stash apply                       # 恢复最近一次暂存（保留记录）
git stash apply stash@{N}             # 恢复指定暂存
git stash drop stash@{N}              # 删除指定暂存
git stash clear                       # 清空所有暂存
git stash show -p                     # 查看暂存内容的 diff
```

**使用场合**: 开发中途需要切换分支处理紧急任务，但当前修改不想提交时。

---

## 九、日志与历史查看

```bash
git log                               # 查看提交日志
git log --oneline                     # 单行简洁格式
git log --graph --oneline --all       # 图形化显示分支历史
git log -N                            # 查看最近 N 条提交
git log --author="name"               # 按作者筛选
git log --since="2024-01-01"          # 按日期筛选
git log -p <file>                     # 查看文件的修改历史
git log --stat                        # 显示每次提交的文件变更统计
git shortlog -sn                      # 按提交次数统计贡献者
```

### 追溯文件变更

```bash
git blame <file>                      # 逐行查看最后修改人和提交
git bisect start                      # 开始二分查找 bug 引入点
git bisect bad <commit>               # 标记有 bug 的提交
git bisect good <commit>              # 标记正常的提交
```

**使用场合**: `bisect` 在大量提交中定位引入 bug 的具体提交，效率远高于逐个检查。

---

## 十、子模块（Submodule）

```bash
git submodule add <url> <path>        # 添加子模块
git submodule init                    # 初始化子模块
git submodule update                  # 更新子模块
git submodule update --init --recursive  # 初始化并递归更新
git submodule foreach git pull        # 批量更新所有子模块
git clone --recursive <url>           # 克隆时自动初始化子模块
```

**使用场合**: 项目依赖另一个 Git 仓库作为子项目（如共享库、文档模板）。

---

## 十一、高级技巧

### Cherry-Pick

```bash
git cherry-pick <commit>              # 拣选单个提交
git cherry-pick <c1> <c2> <c3>        # 拣选多个提交
git cherry-pick <c1>..<c2>            # 拣选范围
git cherry-pick --no-commit <commit>  # 只应用更改，不自动提交
```

**使用场合**: 从一个分支挑选特定提交应用到另一个分支（如热修复同步到多个版本分支）。

### Worktree

```bash
git worktree add <path> <branch>      # 创建工作树
git worktree list                     # 列出工作树
git worktree remove <path>            # 删除工作树
```

**使用场合**: 需要同时在多个分支上工作，不想来回切换和 stash。

### 清理

```bash
git clean -fd                         # 删除未跟踪的文件和目录
git clean -n                          # 预览会被删除的文件（dry-run）
```

### 配置

```bash
git config --global user.name "name"
git config --global user.email "email"
git config --global core.editor "code --wait"
git config --global init.defaultBranch main
git config --global pull.rebase true
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --all"
```

---

## 十二、常见工作流

### Git Flow

```
main        ──●──────────────────●──────────●──
               \                /          /
develop    ─────●───●───●───●───●───●───●──
                \     /       \       /
feature     ─────●───●         ●─────●
```

- `main`: 生产环境代码
- `develop`: 开发主线
- `feature/*`: 功能分支，从 develop 创建，完成后合并回 develop
- `release/*`: 发布准备分支
- `hotfix/*`: 生产环境紧急修复

### GitHub Flow（推荐用于持续部署）

```
main    ──●───────●───────●──
           \     / \     /
feature  ───●───●   ●───●──
```

- 从 main 创建 feature 分支
- 开发完成后提交 PR
- Code Review + CI 通过后合并到 main
- main 随时可部署

### Trunk-Based Development

```
main    ──●──●──●──●──●──●──●──
           \  /    \  /
short    ────●      ●────
```

- 所有开发在主干上进行
- 短生命周期分支（< 1 天），频繁集成
- 配合 Feature Flag 控制功能发布
