#!/bin/bash

# Hexo 清理与生成
echo "🧹 清理旧文件..."
npx hexo clean

echo "🛠️ 生成新网站..."
npx hexo g

echo "🚀 正在部署到 GitHub Pages..."
npx hexo d
