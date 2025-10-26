# Git Cheatsheet for Plant Disease Detection Project

## 🚀 Quick Start

### Push Your Changes (Now Much Faster!)
```bash
git push origin main
```

### Check Status Before Pushing
```bash
git status              # See what's changed
git diff               # See exact changes
git diff --cached      # See staged changes
```

## 📋 Common Workflows

### Adding New Features
```bash
# 1. Check current status
git status

# 2. Make your changes
# (Edit files...)

# 3. Stage changes
git add .

# 4. Commit with a message
git commit -m "Add new feature description"

# 5. Push to GitHub
git push origin main
```

### Creating a Feature Branch (Recommended for Teams)
```bash
git checkout -b feature/new-feature-name
# ... make changes ...
git add .
git commit -m "Describe your changes"
git push origin feature/new-feature-name
# Then create a Pull Request on GitHub
```

## ⚠️ What's Ignored (And Why)

### Virtual Environments
```
venv/, env/, .venv/
```
- **Why:** Recreate with `python -m venv venv` and `pip install -r requirements.txt`

### Python Cache
```
__pycache__/, *.pyc, *.egg-info/
```
- **Why:** Auto-generated, recreated when code runs

### Model Files
```
*.pt, *.pth, *.h5, *.model
```
- **Why:** Too large for git (use Git LFS, cloud storage, or model registries)

### Node Modules
```
node_modules/
```
- **Why:** Recreate with `npm install` or `bun install`

### IDE Files
```
.vscode/, .idea/, *.swp
```
- **Why:** Personal editor settings, shouldn't be shared

## 🛑 DO NOT Commit

❌ Virtual environments (venv)
❌ Large model files
❌ node_modules
❌ Database files (*.db)
❌ API keys or secrets (.env files)
❌ Generated files or caches

## ✅ DO Commit

✅ Source code (.py, .tsx, .ts, .js)
✅ Configuration files (requirements.txt, package.json)
✅ Documentation (README.md, .md files)
✅ Tests (test_*.py)

## 🐛 Troubleshooting

### Accidentally Committed Large File?
```bash
# Remove from tracking (but keep file locally)
git rm --cached large_file.pt
# Add to .gitignore
echo "*.pt" >> .gitignore
# Commit the change
git add .gitignore
git commit -m "Remove large file from tracking"
```

### Undo Last Commit (Before Push)
```bash
git reset --soft HEAD~1
# Or undo changes entirely:
git reset --hard HEAD~1
```

### Check What Will Be Pushed
```bash
git log --oneline -n 5              # See last 5 commits
git log --oneline origin/main..HEAD # See unpushed commits
```

## 📊 Performance Tips

✅ The `.gitignore` file now protects against large files
✅ Model files removed from tracking (210MB savings)
✅ Python cache excluded from tracking
✅ Push should now be much faster!

## 🔗 Useful Resources

- **Git Documentation:** https://git-scm.com/doc
- **GitHub Help:** https://help.github.com/
- **Git LFS:** https://git-lfs.github.com/ (for large files)

---
**Last Updated:** 2025-10-26
