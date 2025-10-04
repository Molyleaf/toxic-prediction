# 1. 基础镜像
FROM python:3.13-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 设置环境变量，避免 apt-get 交互式提示
ARG DEBIAN_FRONTEND=noninteractive

# 4. 设置 Python 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 5. 复制依赖文件
COPY requirements.txt .

# 6. 安装系统依赖
# 先切换为南京大学镜像源以加速下载，然后安装必要的系统库
RUN set -eux; \
    codename="$(. /etc/os-release; echo "$VERSION_CODENAME")"; \
    rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; \
    : > /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s-updates main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian-security %s-security main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s-backports main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 7. 安装 Python 依赖
# 设置 pip 镜像源并安装依赖
RUN pip config set global.index-url https://mirror.nju.edu.cn/pypi/web/simple \
 && pip config set install.trusted-host mirror.nju.edu.cn \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# --- 以下为核心修改 ---

# 8. 创建一个无特权的 appuser 用户 (uid:2000) 和 appgroup 组 (gid:2000)
RUN groupadd -g 2000 appgroup && useradd -r -u 2000 -g appgroup appuser

# 9. 复制应用程序代码，并直接将所有者设置为新创建的 appuser 用户
# 使用 --chown 可以避免额外执行一次 chown 命令，优化了镜像分层
COPY --chown=appuser:appgroup . .

# 10. 切换到 appuser 用户来运行后续的命令
USER appuser

# 11. 声明 Flask 应用运行的端口
EXPOSE 5000

# 12. 定义启动应用的命令 (使用 gunicorn)，此命令将由 appuser 用户执行
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]