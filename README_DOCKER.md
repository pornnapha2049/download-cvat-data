# 🐳 CVAT to YOLO Studio - Docker Guide

คู่มือการ Build และรัน **CVAT to YOLO Studio** ด้วย Docker พร้อมระบบ **Web Browser GUI (noVNC)** ที่ใช้งานได้ทันทีบน Windows, macOS, และ Linux โดยไม่ต้องติดตั้ง X-Server เพิ่มเติม

---

## 🚀 วิธีเริ่มต้นใช้งานแบบรวดเร็ว (Quick Start)

### 1. รันผ่าน Docker Compose (แนะนำที่สุด)

เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วสั่ง:

```bash
# Build และเริ่มรัน Container
docker compose up -d
```

เมื่อ Container รันเสร็จแล้ว สามารถเปิดใช้งานโปรแกรมผ่านเว็บเบราว์เซอร์ได้ทันที:
👉 **[http://localhost:8081](http://localhost:8081)** หรือ **http://1.10.131.222:8081** (สำหรับเครื่องในวง LAN)

---

## 🎮 โหมดการใช้งาน (Usage Modes)

### โหมดที่ 1: ใช้งานผ่าน Web Browser (noVNC) — *ค่าเริ่มต้น*
เหมาะสำหรับผู้ใช้ **Windows / Mac / Linux** ทุกระบบ:
- ไม่ต้องติดตั้ง X11 / VcXsrv ในเครื่อง Host
- หน้าจอ UI จะแสดงผลในเบราว์เซอร์ (Chrome, Edge, Firefox, Safari) แบบเต็มฟังก์ชันพร้อมภาษาไทยและ Sticker Theme
- ปรับขนาดและ Auto-Scale ตามหน้าจอเบราว์เซอร์อัตโนมัติ

```bash
docker compose up -d
```

### โหมดที่ 2: รันด้วย NVIDIA GPU (สำหรับ Train / Inference)
หากเครื่อง Host มีการ์ดจอ NVIDIA และติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) แล้ว:

```bash
# รัน Profile GPU
docker compose --profile gpu up -d
```
หรือใช้คำสั่ง `docker run`:
```bash
docker run -d \
  --name cvat-yolo-studio \
  --gpus all \
  -p 6080:6080 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/runs:/app/runs \
  --shm-size=4gb \
  cvat-yolo-studio:latest
```

### โหมดที่ 3: รันแบบ Native X11 บน Linux / WSLg (Direct Window)
หากต้องการให้หน้าต่างโปรแกรมเด้งขึ้นมาบน Desktop ของ Linux หรือ WSL2 (WSLg) โดยตรง:

```bash
# อนุญาตการเชื่อมต่อ X11 (บน Linux)
xhost +local:root

# สั่งรันโดยปิด Web GUI
docker run -it --rm \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e WEB_GUI=false \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  cvat-yolo-studio:latest
```

---

## 💾 การเก็บรักษาข้อมูล (Data Persistence)

ใน `docker-compose.yml` มีการเชื่อมโยง Volume เข้ากับเครื่อง Host ดังนี้:
* `./data` $\rightarrow$ โฟลเดอร์เก็บ Dataset ที่ดาวน์โหลดและ Merge/Convert/Split
* `./models` $\rightarrow$ โฟลเดอร์เก็บไฟล์ Weights (`.pt`)
* `./runs` $\rightarrow$ โฟลเดอร์เก็บผลลัพธ์การ Train และการทดสอบโมเดล

> **ข้อมูลทุกอย่างจะไม่สูญหาย** แม้จะปิดหรือลบ Container ทิ้งไปก็ตาม

---

## ⚙️ การตั้งค่าสภาพแวดล้อม (Environment Variables)

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :--- | :--- |
| `WEB_GUI` | `true` | เปิดระบบ noVNC สำหรับเข้าใช้งานผ่าน Web Browser (`true` / `false`) |
| `RESOLUTION` | `1600x950` | ความละเอียดหน้าจอเสมือน เช่น `1920x1080`, `1440x900` |
| `WEB_PORT` | `6080` | พอร์ตสำหรับเปิด Web Browser noVNC |
| `VNC_PORT` | `5900` | พอร์ตสำหรับโปรแกรม VNC Client ภายนอก |

---

## 🛠️ คำสั่งที่มีประโยชน์

```bash
# ดู Log การทำงานของแอปพลิเคชัน
docker compose logs -f

# หยุดการทำงาน
docker compose down

# Rebuild อิมเมจใหม่เมื่อมีการอัปเดตโค้ด
docker compose build --no-cache
```
