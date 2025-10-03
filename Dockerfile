FROM python:3.13-slim

# 2. 设置工作目录
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. 复制依赖文件并安装
COPY requirements.txt .

# 4. 安装依赖
# 先用官方源安装 CA 证书，再切换为南京大学镜像并更新
RUN set -eux; \
    # apt-get update; \
    # apt-get install -y --no-install-recommends ca-certificates; \
    # rm -rf /var/lib/apt/lists/*; \
    codename="$(. /etc/os-release; echo "$VERSION_CODENAME")"; \
    rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; \
    : > /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s-updates main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian-security %s-security main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    printf "deb https://mirrors.nju.edu.cn/debian/ %s-backports main contrib non-free non-free-firmware\n" "$codename" >> /etc/apt/sources.list; \
    apt-get update


# 后续正常 apt 安装
RUN apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# 5. 安装pip
RUN pip config set global.index-url https://mirror.nju.edu.cn/pypi/web/simple \
 && pip config set install.trusted-host mirror.nju.edu.cn \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# 6. 复制应用程序代码和预训练模型到容器中
COPY . .

# 7. 声明 Flask 应用运行的端口
EXPOSE 5000

# 8. 定义启动应用的命令 (使用 gunicorn)
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]