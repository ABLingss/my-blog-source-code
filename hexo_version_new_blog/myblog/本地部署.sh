#!/bin/bash

# 确保 hexo 环境正确
echo "🔧 正在清理旧文件..."
hexo clean

echo "🛠️ 正在生成网站..."
hexo g

echo "🚀 启动本地服务器..."
hexo s
