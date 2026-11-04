# 🚀 CVAT to YOLO Studio & Auto-Trainer (Standalone AI Pipeline)

โปรแกรม Python GUI (PyQt5) สำหรับประมวลผล Computer Vision แบบ **All-in-One Standalone Pipeline (7 ขั้นตอน)**:
1. **CVAT Login**: หน้าเข้าสู่ระบบ CVAT Server แยกอิสระ ปลอดภัย ไม่มี URL เริ่มต้นค้างไว้
2. **CVAT Downloader**: กรองและเลือก Task ดาวน์โหลด Annotation และรูปภาพ (แตกไฟล์และย้ายรูปออกจากโฟลเดอร์ `default` อัตโนมัติ พร้อมลบโฟลเดอร์ `default` ทิ้ง)
3. **Merge Datasets**: รวม Dataset หลายๆ Task เข้าด้วยกันในโฟลเดอร์ `data/` (Merge & MD5 Deduplication)
4. **COCO to YOLO Converter**: แปลงฟอร์แมต COCO 1.0 เป็น YOLO (Detection Bounding Box หรือ Segmentation Polygon)
5. **Dataset Splitter**: แบ่งข้อมูล Train / Val / Test (ครอบคลุมทุกรูป 100% ทั้งที่มี Label และไม่มี Label/รูป Background)
6. **YOLO Auto-Trainer**: Train โมเดล YOLO ด้วยระบบ Auto-Tune Hyperparameters, Early Stopping (Patience), Run Versioning (`_v1`, `_v2`, ...) และเปลี่ยนชื่อ `best.pt` เป็น `{custom_name}_v{num}.pt` ใน `runs/.../weights/` โดยตรง
7. **Test Model (Inference & Evaluation)**: ทดสอบโมเดลที่ Train แล้วกับภาพเดี่ยว (Single Image), ทั้งโฟลเดอร์ (Entire Folder), หรือไฟล์วิดีโอ (Video File) พร้อมดูผลลัพธ์พรีวิวแบบ Real-time และเลือกบันทึกผลได้

---

## 🌟 ฟีเจอร์เด่น (Key Features)

1. **หน้าจอแยกตามขั้นตอนแบบ Sequential Standalone (7 Steps Flow)**:
   - นำทางลื่นไหลทีละขั้นตอนแบบ Step-by-Step ผ่านปุ่ม **"Next Step ➔"** และ **"< Previous"** โดยไม่มีปุ่มแท็บใหญ่เกะกะด้านบน
   - หน้า **Login ถูกแยกเป็นหน้าแรก (Step 1)** อย่างชัดเจน พร้อมการ์ดแสดงสถานะและข้อมูล Server เมื่อเชื่อมต่อสำเร็จ
   - แถบด้านบนออกแบบตามสไตล์ Mobile Wellness App ประกอบด้วยปุ่มย้อนกลับวงกลม `< Back`, หัวข้อหน้าจอปัจจุบัน, กระดิ่ง Notification สีกรมท่าพร้อมจุดแจ้งเตือนสีชมพู, และมาสคอต Clover Byte
   - แถบด้านล่างมีตัวบอกสถานะ Carousel Step Pills (1 ถึง 7) สำหรับติดตามและสลับไปยังหน้าที่ต้องการได้อย่างยืดหยุ่น
   - แต่ละหน้ามีตัวเลือก Input / Output และปุ่มประมวลผลของตนเองอย่างสมบูรณ์ และมีปุ่มกดก้าวไปสู่ขั้นตอนถัดไปในตัว

2. **รวมไฟล์ข้อมูลทั้งหมดไว้ในโฟลเดอร์ `data/` เดียว (Single Unified Data Folder)**:
   - ไม่มีการสร้างโฟลเดอร์กระจัดกระจาย ข้อมูลภาพและ Label ทั้งหมดจัดเก็บอย่างเป็นระเบียบภายใต้โฟลเดอร์ `C:\Articulus\cvat\data`:
     - `data/raw_tasks/`: เก็บไฟล์ดิบของ Tasks ที่ดาวน์โหลดมาจาก CVAT
     - `data/images/`: เก็บไฟล์รูปภาพทั้งหมด (รวมถึงรูปภาพที่แบ่ง Train, Val, Test)
     - `data/labels/`: เก็บไฟล์ Label `.txt` ของ YOLO (แบ่ง Train, Val, Test)
     - `data/annotations/`: เก็บไฟล์ `instances_default.json` ของ COCO 1.0
     - `data/test_results/`: เก็บผลลัพธ์ภาพและวิดีโอจากการทดสอบ Model Inference
     - `data/data.yaml`: ไฟล์คอนฟิกสำหรับใช้ Train โมเดล YOLO
   - มีปุ่ม **"Open data/ Folder"** ที่แถบ Header เพื่อเปิดดูข้อมูลใน Windows File Explorer ได้ตลอดเวลา

3. **ไม่มี URL เริ่มต้น (No Default URL)**:
   - ช่องกรอก **CVAT Server URL** เริ่มต้นเป็นช่องว่าง พร้อมตัวอย่าง Placeholder เพื่อให้ผู้ใช้กรอกหรือแปะ URL เซิร์ฟเวอร์ของตนเองได้อย่างอิสระและปลอดภัย

4. **CVAT Connect & Smart Task Downloader (Standalone)**:
   - เชื่อมต่อ CVAT Server และดึงรายการ Project / Tasks พร้อมข้อมูล Stage, State, Status, จำนวนภาพ
   - ตัวกรองสถานะ: Filter ตาม Project, Stage (`acceptance`, `annotation`, `validation`), State (`completed`, `in progress`, `new`, `rejected`), Status, และค้นหาตามชื่อ Task หรือเลือก Checkbox ทีละอันได้อิสระ
   - ดาวน์โหลดในฟอร์แมต **COCO 1.0** พร้อมภาพ และแตกไฟล์ ZIP อัตโนมัติ
   - **นำรูปออกจากโฟลเดอร์ `/images/default/` มาไว้ที่ `/images/` แล้วลบโฟลเดอร์ `default` ทิ้งทันที**

5. **Dataset Merger (รวมรูปภาพและ Annotations)**:
   - รวมทุก Task ที่ดาวน์โหลดมาเข้าเป็น Unified COCO Dataset ภายในโฟลเดอร์ `data/`
   - ระบบป้องกันชื่อไฟล์ภาพซ้ำกันข้าม Task ด้วย MD5 Collision Resolver
   - ปรับ Category ID และ Annotation ID ใหม่ให้อัตโนมัติ

6. **COCO 1.0 to YOLO Converter**:
   - รองรับ 2 โหมด:
     - 🎯 **Detection (`detect`)**: แปลง Annotation จาก Segmentation Polygon เป็น Bounding Box (`cls xc yc w h` แบบ Normalized)
     - 🔷 **Segmentation (`segment`)**: รักษาพิกัด Polygon ละเอียด (`cls x1 y1 x2 y2 ...` แบบ Normalized)
   - สร้างไฟล์ Label `.txt` ให้ทุกภาพไว้ใน `data/labels/`

7. **Train / Val / Test Splitter**:
   - ปรับสัดส่วนเปอร์เซ็นต์ได้อิสระ เช่น Train 70% / Val 20% / Test 10%
   - **นำทุกรูปเข้าร่วมการ Train (รวมถึงภาพที่ไม่มี Label หรือ Background Images จะได้รับไฟล์ Label ว่างตามมาตรฐาน Ultralytics YOLO)**
   - สร้างไฟล์คอนฟิก `data.yaml` อัตโนมัติด้วย Absolute Path

8. **YOLO Auto-Trainer & Clean Weights Management**:
   - รองรับทั้งงาน Detection และ Segmentation ตระกูล **YOLO11, YOLOv8, YOLO26 (26n, 26s, 26m, 26l)** หรือ Custom `.pt`
   - **Auto-Tune Hyperparameters**: คำนวณ Epochs, Batch Size, Img Size, Learning Rate และ Early Stopping (Patience) ที่เหมาะสมกับขนาดข้อมูล
   - **Early Stopping (Patience)**: หยุด Train ทันทีเมื่อค่า Loss/mAP นิ่ง
   - **Auto-Versioning & Weights Naming**:
     - บันทึกการทดลองไว้ใน `runs/{task}/{custom_name}_v{run_num}/`
     - **ไม่มีโฟลเดอร์ weights แยกภายนอก**: ภายในโฟลเดอร์ `runs/{task}/{custom_name}_v{run_num}/weights/` โปรแกรมจะเปลี่ยนชื่อ `best.pt` เป็น `{custom_name}_v{run_num}.pt` โดยตรง พร้อมคงไฟล์ `last.pt` ไว้

9. **Test Model / Inference (ทดสอบโมเดลกับภาพ โฟลเดอร์ หรือวิดีโอ)**:
   - เลือกระหว่าง:
     - 🖼️ **Single Image**: เลือกลองภาพเดี่ยว พร้อมดูภาพผลลัพธ์และตารางพิกัด Bounding Box ทันที
     - 📁 **Entire Images Folder**: ทดสอบทุกภาพในโฟลเดอร์แบบ Batch พร้อมปุ่ม Carousel `< Prev` / `Next >` เปิดดูภาพผลลัพธ์ทีละภาพ
     - 🎥 **Video File**: ประมวลผลและเล่น Preview เฟรมวิดีโอแบบสด
   - ปรับ Confidence Threshold และ IoU Threshold ได้อิสระ
   - **ตัวเลือกบันทึกผลลัพธ์ (Save Prediction Results)**: เลือกว่าจะบันทึกภาพ/วิดีโอผลลัพธ์ที่วาดกรอบแล้วไว้ใน `data/test_results/` หรือไม่

10. **ดีไซน์ Mindfulness Pastel Wellness พร้อม SVG Vector แท้ 100%**:
    - ออกแบบตามโทนสีสุขภาพจิตและการเรียนที่สบายตา ([Pinterest Reference](https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg))
    - คลังไอคอน SVG 37 แบบใน [`ui/svg_icons.py`](file:///C:/Articulus/cvat/ui/svg_icons.py) คมชัดทุกขนาดหน้าจอ
    - ตกแต่งด้วย **Dancing SVG Mascot ("Clover Byte" .☘︎ ݁˖)** ดิ้นโบกมือและตอบสนองตามสถานะการทำงาน

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
C:\Articulus\cvat\
├── app.py                     # ตัวเปิดโปรแกรมหลัก (GUI Launcher)
├── requirements.txt           # รายการแพ็กเกจที่ต้องใช้งาน
├── data/                      # โฟลเดอร์รวมข้อมูล Dataset ทั้งหมดเพียงโฟลเดอร์เดียว
│   ├── raw_tasks/             # Tasks ดิบที่ดาวน์โหลดมาจาก CVAT
│   ├── images/                # รูปภาพทั้งหมด (และ images/train, val, test)
│   ├── labels/                # ไฟล์ Label .txt (และ labels/train, val, test)
│   ├── annotations/           # COCO 1.0 instances_default.json
│   ├── test_results/          # ผลลัพธ์ภาพ/วิดีโอจากการทดสอบโมเดล
│   └── data.yaml              # ไฟล์คอนฟิกสำหรับ Train YOLO
├── runs/                      # ผลลัพธ์ Checkpoints, Graphs และ Weights จากการ Train
│   └── detect/ (หรือ segment/)
│       └── my_model_v1/
│           └── weights/
│               ├── my_model_v1.pt  # best.pt ที่ถูกเปลี่ยนชื่อเป็น custom name
│               └── last.pt
├── core/
│   ├── cvat_client.py         # เชื่อมต่อ CVAT, กรอง Task, ดาวน์โหลด, Unzip & จัดการลบ /images/default
│   ├── merger.py              # รวม COCO JSON และรูปภาพ, ป้องกันชื่อไฟล์ซ้ำด้วย MD5
│   ├── converter.py           # แปลง COCO 1.0 เป็น YOLO (Detect / Segment)
│   ├── splitter.py            # แบ่ง Train/Val/Test (รวมทุกรูปทั้ง labeled และ background)
│   ├── trainer.py             # YOLO Training engine, Auto-tuning, Patience, Run versioning
│   └── tester.py              # YOLO Inference Engine สำหรับ Image, Folder และ Video
└── ui/
    ├── svg_icons.py           # คลังไอคอน SVG เวกเตอร์แท้ สไตล์ Mindfulness & Pastel Wellness
    ├── dancing_svg.py         # SVG Mascot ดิ้นได้ & Animated Activity Pills
    ├── theme.py               # ธีม Mindfulness & Pastel Wellness สบายตา
    └── main_window.py         # หน้าต่างหลัก รวม 7 หน้าการทำงานแบบ Sequential Standalone
```

---

## 🚀 วิธีการติดตั้งและรันโปรแกรม (How to Run)

1. ติดตั้ง Library ที่จำเป็น (หากยังไม่ได้ติดตั้ง):
```powershell
pip install -r requirements.txt
```

2. รันโปรแกรม:
```powershell
cd C:\Articulus\cvat
python app.py
```

---

## 📖 คู่มือการใช้งานทีละขั้นตอน (Step-by-Step Workflow)

### 1. ดาวน์โหลดจาก CVAT (`1. CVAT Downloader`)
1. กรอก **CVAT Server URL** (ช่องเริ่มต้นว่างไว้ ปลอดภัย ไม่ล็อค URL)
2. ใส่ **Username** และ **Password** แล้วกด **"Connect & Fetch"**
3. กรอง Tasks ด้วย Project, Stage (`acceptance`), State (`completed`), หรือค้นหาชื่อ Task
4. ติ๊กเลือก Tasks ที่ต้องการดาวน์โหลด
5. โฟลเดอร์ปลายทางจะตั้งค่าอัตโนมัติไปที่ `C:\Articulus\cvat\data\raw_tasks`
6. กด **"Download & Auto-Extract Tasks (COCO 1.0)"**
   - โปรแกรมแตกไฟล์ Zip อัตโนมัติ
   - ย้ายรูปออกจาก `images/default/*` มาไว้ใน `images/*`
   - ลบโฟลเดอร์ `default` ทิ้งอย่างสะอาดหมดจด

### 2. รวม Dataset (`2. Merge Datasets`)
1. โฟลเดอร์ต้นทางชี้ไปที่ `data/raw_tasks` และปลายทางชี้ไปที่ `data/`
2. กด **"Merge Datasets"** เพื่อรวมทุก Task เข้าด้วยกัน
   - ระบบแก้ปัญหาชื่อไฟล์ซ้ำด้วย MD5 hash อัตโนมัติ
   - รวมรูปภาพไว้ใน `data/images/` และ Annotations ไว้ใน `data/annotations/instances_default.json`

### 3. แปลงเป็นฟอร์แมต YOLO (`3. COCO to YOLO`)
1. ไฟล์ COCO JSON ชี้ไปที่ `data/annotations/instances_default.json`
2. เลือกโหมด:
   - **Detection**: แปลง Polygon เป็น Bounding Box แบบ normalized (`cls xc yc w h`)
   - **Segmentation**: รักษาพิกัด Polygon ละเอียด (`cls x1 y1 x2 y2 ...`)
3. โฟลเดอร์ปลายทางชี้ไปที่ `data/labels/`
4. กด **"Convert COCO to YOLO (.txt)"**

### 4. แบ่งสัดส่วน Train / Val / Test (`4. Dataset Splitter`)
1. โฟลเดอร์ Images และ Labels ชี้ไปที่ `data/images` และ `data/labels`
2. ปรับสัดส่วน Train, Val, Test (เช่น 70% / 20% / 10%)
3. รูปภาพทั้งหมด 100% จะถูกนำมาใช้งาน (รูปที่ไม่มี Label จะมีไฟล์ `.txt` ว่างสำหรับ Background Training)
4. กด **"Split Dataset & Generate data.yaml"** (สร้างไฟล์ `data/data.yaml` อัตโนมัติ)

### 5. เทรนโมเดล YOLO (`5. YOLO Auto-Trainer & Explorer`)
1. เลือก Task: `detect` หรือ `segment`
2. เลือกรุ่นโมเดล: `yolo11n.pt`, `yolov8n.pt`, `yolo26n.pt` หรือ Custom `.pt`
3. ไฟล์ `data.yaml` ชี้ไปที่ `data/data.yaml`
4. กด **"Auto-Tune Hyperparameters for Dataset"** เพื่อให้ระบบคำนวณ Epochs, Batch, Patience อัตโนมัติ
5. ตั้งชื่อใน **"Custom Run Name"** (เช่น `my_yolo_model`)
6. กด **"Start YOLO Training"**
   - ดู Epoch, Loss, mAP และ Live Terminal Logs แบบ Realtime
   - ไฟล์ Weights ที่ดีที่สุดจะถูกคัดลอกเป็น `weights/my_yolo_model.pt`
7. สามารถกด **"Open data/ Folder"** หรือ **"Open runs/ Folder"** เพื่อเปิดดูไฟล์ผลลัพธ์ใน Windows ได้ทันที
