# 🚀 CVAT to YOLO Studio & Auto-Trainer (Enterprise AI Pipeline)

โปรแกรม Python GUI (PyQt5) สำหรับดาวน์โหลด Annotation และรูปภาพจาก **CVAT** (`http://1.10.131.222:8080/projects`), ทำการแตกไฟล์และย้ายรูปออกจากโฟลเดอร์ `default` อัตโนมัติ, รวม Dataset (Merge), แปลงฟอร์แมต COCO 1.0 เป็น YOLO (Detection & Segmentation), แบ่งข้อมูล Train / Val / Test (ครอบคลุมทุกรูปทั้งที่มี Label และไม่มี Label/รูป Background), และ Train โมเดล YOLO ด้วยระบบ Auto-Tune Hyperparameters, Early Stopping (Patience), พร้อมระบบ Run Versioning (`_v1`, `_v2`, ...) และบันทึก Weights ชื่อตามโฟลเดอร์ที่ตั้ง พร้อมตกแต่งด้วย **SVG Mascot ดิ้นได้ (Animated SVG Mascot)** สุดทันสมัย!

---

## 🌟 ฟีเจอร์เด่น (Key Features)

1. **หน้าจอ Login แยกเป็นสัดส่วน (Dedicated Authentication Screen)**:
   - เปิดโปรแกรมมาจะพบกับหน้า **Sign-In เข้าสู่ระบบ CVAT** สวยงามทันสมัย (Cyber Glassmorphism Card)
   - มี **Dancing SVG Mascot ("Byte")** ต้อนรับ ดิ้นโบกมือและกระพริบตาเรืองแสง
   - กรอก **CVAT Server URL**, **Username** และ **Password** พร้อมระบบตรวจสอบสถานะการเชื่อมต่อ
   - มีปุ่ม **"📁 Work with Local Datasets (Skip Login / Offline Mode)"** สำหรับกรณีที่มี Dataset อยู่แล้วและต้องการข้ามไปใช้งาน Merge/Convert/Split/Train ได้ทันทีโดยไม่ต้องล็อกอิน
   - ในหน้า Studio มีปุ่ม **"🚪 Logout"** ที่แถบ Header เพื่อสลับผู้ใช้หรือเปลี่ยน Server ได้ตลอดเวลา

2. **CVAT Connect & Smart Task Downloader**:
   - ดึงรายการ Project และ Tasks พร้อมแสดงข้อมูล Stage, State, Status, จำนวนภาพ
   - ตัวกรองสถานะ: Filter ตาม Project, Stage (เช่น `acceptance`, `annotation`, `validation`), State (เช่น `completed`, `in progress`, `new`, `rejected`), Status, และค้นหาตามชื่อ Task หรือเลือก Checkbox ทีละอันได้อิสระ
   - ดาวน์โหลดในฟอร์แมต **COCO 1.0** พร้อมภาพ
   - **แตกไฟล์ ZIP อัตโนมัติ**
   - **นำรูปออกจากโฟลเดอร์ `/images/default/` มาไว้ที่ `/images/` แล้วลบโฟลเดอร์ `default` ทิ้งทันทีตามคำขอ**

2. **Dataset Merger (รวมรูปภาพและ Annotations ทั้งหมด)**:
   - รวมทุก Task ที่ดาวน์โหลดมาเข้าเป็น Unified COCO Dataset
   - ระบบป้องกันชื่อไฟล์ภาพซ้ำกันข้าม Task ด้วย MD5 Collision Resolver
   - ปรับ Category ID และ Annotation ID ใหม่ให้อัตโนมัติ

3. **COCO 1.0 to YOLO Converter**:
   - รองรับ 2 โหมด:
     - 🎯 **Detection (`detect`)**: แปลง Annotation จาก Segmentation Polygon เป็น Bounding Box (`cls xc yc w h` แบบ Normalized) อัตโนมัติตามโจทย์
     - 🔷 **Segmentation (`segment`)**: รักษาพิกัด Polygon ละเอียด (`cls x1 y1 x2 y2 ...` แบบ Normalized)
   - สร้างไฟล์ Label `.txt` ให้ทุกภาพ

4. **Train / Val / Test Splitter**:
   - ปรับสัดส่วนเปอร์เซ็นต์ได้ตามใจชอบ เช่น Train 70% / Val 20% / Test 10%
   - **นำทุกรูปเข้าร่วมการ Train (รวมถึงภาพที่ไม่มี Label หรือ Background Images จะถูกสร้างเป็นไฟล์ Label ว่างตามมาตรฐาน Ultralytics YOLO)**
   - สร้างไฟล์คอนฟิก `data.yaml` อัตโนมัติด้วย Absolute Path พร้อมใช้งาน

5. **YOLO Auto-Trainer & Early Stopping**:
   - รองรับทั้งงาน Detection และ Segmentation
   - รองรับโมเดลตระกูล **YOLO11, YOLOv8, YOLO26 (26n, 26s, 26m, 26l)** หรือเลือกไฟล์ `.pt` ของตนเองได้
   - **Auto-Tune Hyperparameters**: คำนวณ Epochs, Batch Size, Img Size, Learning Rate และ Early Stopping (Patience) ที่เหมาะสมกับจำนวนภาพและคลาสที่มีให้อัตโนมัติ พร้อมให้ User แก้ไขได้ทุกค่า
   - **Early Stopping (Patience)**: หยุด Train ทันทีเมื่อค่า Loss/mAP นิ่งหรือไม่ดีขึ้นตามจำนวน Epochs ที่กำหนด
   - **Auto-Versioning Folder Naming**: บันทึกโฟลเดอร์ตามชื่อที่ User ตั้ง เช่น `runs/detect/my_model_v1`, ถ้ารันชื่อเดิมซ้ำจะขยับเป็น `_v2`, `_v3`, `_v4` อัตโนมัติ
   - **Custom Named Best Weights**: บันทึกและคัดลอกไฟล์ `best.pt` เป็น `{custom_name}.pt` ไว้ในโฟลเดอร์ `weights/` ทันทีหลัง Train เสร็จ
   - ปุ่ม Stop Training สำหรับสั่งหยุด Train อย่างปลอดภัยได้ตลอดเวลา

6. **ดีไซน์สไตล์ Mindfulness, Pastel Wellness & Study Stationery (SVG ทุกไอคอน)**:
   - ออกแบบตามสไตล์ UI สุขภาพจิตและการเรียนที่สบายตา อิงตาม [Pinterest Design](https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg):
     - โทนสีสงบ ละมุนตา: **Soft Slate-Cream (#f0f4f8)**, **Crisp White Cards (#ffffff)**, **Deep Slate Navy (#132c3f)**, **Matcha Mint (#daf0dd)**, **Lavender Rose (#fde4ed)** และ **Butter Yellow (#fef3d1)**
     - การ์ดขอบมนนุ่มนวล (16px - 20px) พร้อมเงาบางเบาและปุ่มทรง Pill ทันสมัย
   - **ไอคอนทุกตัวในระบบเขียนเป็น SVG เวกเตอร์แท้ 100% (`ui/svg_icons.py`)**:
     - เส้นขอบคมชัด สไตล์มินิมอล ลายเส้นโค้งมน พร้อมเฉดสีพาสเทล
     - ออกแบบไอคอนเฉพาะทาง: **Lotus in Bowl**, **Zen Balance Stones**, **Mind Detox Brain**, **Gratitude Sunburst**, **Notification Bell**, **Bar Charts**, **Circular Arrow Pills**, **Cloud Download**, **Merge Streams**, **Magic Convert Wand**, **Split Cards**, **Rocket Trainer**
     - รองรับการเรนเดอร์ High-DPI คมชัดทุกขนาดหน้าจอ พร้อมระบบ In-memory Caching
   - แบ่งขั้นตอนการทำงานออกเป็น **5 หน้า (Steps)** อย่างชัดเจน:
     - `1. Task Explorer` ➔ `2. Merge & Convert` ➔ `3. Dataset Splitter` ➔ `4. YOLO Auto-Trainer` ➔ `5. Runs & Explorer`
   - **Interactive Stepper Bar**: ด้านบนแสดงปุ่มขั้นตอนทั้ง 5 พร้อมไอคอน SVG ประจำแต่ละขั้นตอน
   - ตกแต่งด้วย **Dancing SVG Mascot ("Clover Byte" .☘︎ ݁˖)** ดิ้นโบกมือและเปลี่ยนท่าทางตามสถานะการทำงานแบบเรียลไทม์

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
C:\Articulus\cvat\
├── app.py                     # ตัวเปิดโปรแกรมหลัก (GUI Launcher)
├── requirements.txt           # รายการแพ็กเกจที่ต้องใช้งาน
├── core/
│   ├── __init__.py
│   ├── cvat_client.py         # เชื่อมต่อ CVAT, กรอง Task, ดาวน์โหลด, Unzip & จัดการลบ /images/default
│   ├── merger.py              # รวม COCO JSON และรูปภาพ, ป้องกันชื่อไฟล์ซ้ำ
│   ├── converter.py           # แปลง COCO 1.0 เป็น YOLO (Detect / Segment)
│   ├── splitter.py            # แบ่ง Train/Val/Test (รวมทุกรูปทั้ง labeled และ background)
│   └── trainer.py             # YOLO Training engine, Auto-tuning, Patience, Run versioning
└── ui/
    ├── __init__.py
    ├── svg_icons.py           # คลังไอคอน SVG เวกเตอร์แท้ สไตล์ Mindfulness & Pastel Wellness
    ├── dancing_svg.py         # SVG Mascot ดิ้นได้ & Animated Activity Pills
    ├── theme.py               # ธีม Mindfulness & Pastel Wellness สบายตา
    └── main_window.py         # หน้าต่างหลัก รวม 5 แท็บการทำงานอย่างเป็นระบบ
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

### ขั้นตอนที่ 1: ดาวน์โหลดจาก CVAT (`🌐 1. CVAT Connect & Download`)
1. กรอก URL (ค่าเริ่มต้นคือ `http://1.10.131.222:8080/projects`)
2. ใส่ **Username** และ **Password** ของ CVAT
3. กด **"🔑 Connect & Fetch Tasks"**
4. สามารถเลือกฟิลเตอร์:
   - **Project**: เลือกเฉพาะ Project ที่ต้องการ
   - **Stage**: กรองตาม `acceptance`, `annotation`, หรือ `validation`
   - **State**: กรองตาม `completed`, `in progress`, `new`, หรือ `rejected`
   - หรือค้นหาชื่อ Task และติ๊ก Checkbox เลือกเองได้อิสระ
5. เลือกโฟลเดอร์ปลายทาง (ค่าเริ่มต้น `C:\Articulus\cvat\downloads`)
6. กด **"⬇️ Download & Auto-Extract Tasks (COCO 1.0)"**
   - โปรแกรมจะดาวน์โหลดไฟล์ Zip
   - แตกไฟล์ Zip อัตโนมัติ
   - ย้ายรูปทั้งหมดจาก `images/default/*` มาไว้ใน `images/*`
   - ลบโฟลเดอร์ `default` ทิ้งอย่างสะอาดหมดจด

### ขั้นตอนที่ 2: รวมไฟล์และแปลงฟอร์แมต (`🔄 2. Merge & Convert`)
1. ตรวจสอบโฟลเดอร์ Tasks ที่ดาวน์โหลดมา
2. กด **"🧩 Merge Datasets"** เพื่อรวมทุก Task เข้าด้วยกัน (แก้ปัญหา Category และชื่อภาพซ้ำให้อัตโนมัติ)
3. เลือกโหมด YOLO:
   - **🎯 Detection**: โปรแกรมจะแปลง Segmentation Polygon หรือ Bounding Box ให้อยู่ในรูป Bounding Box กล่องสี่เหลี่ยมแบบ normalized (`cls xc yc w h`)
   - **🔷 Segmentation**: รักษาเส้นขอบ Polygon แบบ normalized (`cls x1 y1 x2 y2 ...`)
4. กด **"⚡ Convert COCO to YOLO (.txt)"**

### ขั้นตอนที่ 3: แบ่ง Dataset Train / Val / Test (`✂️ 3. Dataset Splitter`)
1. ปรับแถบเลื่อนสัดส่วน Train, Val, Test (เช่น 70% / 20% / 10%)
2. ตัวเลือก **"Include ALL images in dataset"** จะเปิดอยู่เสมอ เพื่อนำทุกรูปเข้าร่วม (รูปที่ไม่มี Label จะมีไฟล์ `.txt` ว่างสำหรับ Negative Background Training)
3. กด **"📦 Split Dataset & Generate data.yaml"**

### ขั้นตอนที่ 4: Train โมเดล YOLO (`🧠 4. YOLO Auto-Trainer`)
1. เลือก Task: `detect` หรือ `segment`
2. เลือกรุ่นโมเดล: `yolo11n.pt`, `yolov8n.pt`, `yolo26n.pt` หรือ Custom `.pt`
3. กดปุ่ม **"⚡ Auto-Tune Hyperparameters for Dataset"** เพื่อให้โปรแกรมคำนวณ Epochs, Batch, Patience, LR ที่เหมาะสมกับขนาด Dataset ให้โดยอัตโนมัติ
4. ปรับแก้ค่า Hyperparameters ตามความต้องการ:
   - **Early Stop (Patience)**: กำหนดจำนวน Epochs ที่ไม่มีการพัฒนาแล้วจะหยุด Train ทันที
5. ตั้งชื่อโฟลเดอร์ที่ช่อง **"Custom Run Name"** (เช่น `inspection_model`)
   - โปรแกรมจะสร้างโฟลเดอร์ `runs/detect/inspection_model_v1` (หรือ `_v2`, `_v3` หากชื่อซ้ำ)
   - ไฟล์ Weights ที่ดีที่สุดจะถูกคัดลอกเป็น `runs/detect/inspection_model_v1/weights/inspection_model.pt`
6. กดปุ่ม **"🚀 Start YOLO Training"**
   - ดูความคืบหน้า Epoch, Loss, Best Fitness, และ Live Terminal Logs ได้แบบ Realtime
   - สามารถกดปุ่ม **"⏹️ Stop Training"** เพื่อหยุดกระบวนการได้ตลอดเวลา

### ขั้นตอนที่ 5: สำรวจผลลัพธ์ (`📂 5. Runs & Explorer`)
- กดปุ่มเปิดโฟลเดอร์ต่างๆ ใน Windows File Explorer ได้ทันที
- ดูไฟล์คอนฟิก `data.yaml` และ Weights ที่บันทึกไว้
