# Dataset Sharing Guide for Colleagues
# RT-MonoDepth-Construction Benchmark Datasets

## 📊 **Current Dataset Inventory**

Based on your downloaded datasets:

```
datasets/                              Total: 31 GB
├── nyu_depth_v2/                     2.8 GB  ✅
│   └── nyu_depth_v2_labeled.mat
├── kitti/                            13 GB   ✅
│   ├── data_depth_annotated/
│   └── raw_data_downloader/
├── cityscapes/                       14 GB   ✅
│   ├── leftImg8bit_trainvaltest/
│   ├── camera_trainvaltest/
│   └── disparity_trainvaltest/
└── make3d/                           700 MB  ✅
    ├── *.jpg (images)
    └── *.dat (depth files)
```

---

## 🚀 **RECOMMENDED SHARING METHODS**

### **Option 1: Cloud Storage (BEST for Colleagues) ⭐⭐⭐**

#### **A. Google Drive (Institutional Account)**
**Pros:** 
- Unlimited storage (with edu account)
- Easy sharing with links
- Built-in compression
- Versioning
- No technical setup

**Cons:**
- Initial upload time (8-12 hours for 59GB)
- Requires Google account

**How to Share:**
```bash
# 1. Create compressed archives (optional but recommended)
cd /Users/ashwani/Desktop/RT-Monodepth-Construction

# Compress each dataset separately (parallel compression)
tar -czf datasets_nyu.tar.gz datasets/nyu_depth_v2/ &
tar -czf datasets_kitti.tar.gz datasets/kitti/ &
tar -czf datasets_cityscapes.tar.gz datasets/cityscapes/ &
tar -czf datasets_make3d.tar.gz datasets/make3d/ &
wait

# 2. Upload to Google Drive (via web or Drive Desktop)
# 3. Share link with "Viewer" permission
```

**Storage Required:** ~28 GB compressed (compression ratio ~10%)

**Upload Time Estimate:**
- With 50 Mbps upload: ~1.5 hours
- With 100 Mbps upload: ~45 minutes
- With 1 Gbps: ~5 minutes

---

#### **B. OneDrive for Business / Microsoft 365 (If Available)**
Similar to Google Drive, often comes with university/institution subscription.

**Pros:**
- 1TB+ storage per user
- Direct Windows integration
- Good sharing features

**Setup:**
Same as Google Drive - compress and upload.

---

#### **C. Institutional File Share / NAS Server**
If your university/lab has an internal file server:

**Pros:**
- Fast LAN transfer (1-10 Gbps)
- No external upload needed
- Already trusted infrastructure

**How to Share:**
```bash
# Copy to shared network drive
rsync -avP --compress datasets/ /Volumes/LabNAS/RT-Monodepth-Datasets/

# Or use SMB/CIFS mount
cp -r datasets/ /mnt/lab-share/RT-Monodepth-Datasets/
```

---

### **Option 2: Academic Cloud Storage (BEST for Research) ⭐⭐⭐**

#### **A. Zenodo (Open Science Repository)**
**Website:** https://zenodo.org/

**Pros:**
- Free for research datasets (up to 50 GB per dataset)
- DOI assigned (citable!)
- Long-term preservation
- Version control
- Perfect for publication supplementary materials

**Cons:**
- 50 GB limit per upload (need to split)
- Public by default (can be embargoed)

**How to Use:**
```bash
# 1. Create account at https://zenodo.org
# 2. Create new upload
# 3. Split datasets if needed:
#    - Upload 1: NYU + Make3D (3.5 GB)
#    - Upload 2: KITTI (27 GB)
#    - Upload 3: Cityscapes (29 GB)
# 4. Get DOI and share with colleagues
```

**HUGE BENEFIT:** When you publish your paper, you can cite your own dataset collection!
```
@dataset{your_name_2025,
  author = {Your Name},
  title = {RT-MonoDepth Construction Site Benchmark Datasets},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

---

#### **B. IEEE DataPort (For IEEE Submissions)**
**Website:** https://ieee-dataport.org/

**Pros:**
- Free for IEEE members
- Up to 2TB storage
- DOI assigned
- IEEE branding (credibility)

**Cons:**
- Need IEEE membership
- Slower than Zenodo

---

#### **C. figshare (Alternative to Zenodo)**
**Website:** https://figshare.com/

**Pros:**
- 20 GB free per file
- Unlimited public storage
- DOI for citations
- Version control

---

### **Option 3: Direct Transfer (BEST for Local Team) ⭐⭐**

#### **A. External Hard Drive (Fastest for Same Location)**
**Recommended:** Samsung T7 / T9 (2TB SSD) - $150-200

**Pros:**
- Fastest method (500+ MB/s)
- No internet needed
- Reusable
- Works offline

**How to Share:**
```bash
# Format as exFAT (cross-platform)
# Copy datasets (takes ~2 minutes for 59GB)
cp -r datasets/ /Volumes/ExternalDrive/RT-Monodepth-Datasets/

# Hand to colleague
```

**Time:** 2-5 minutes copy time per colleague

---

#### **B. Syncthing (Peer-to-Peer Sync)**
**Website:** https://syncthing.net/

**Pros:**
- No central server needed
- Encrypted P2P transfer
- Automatic sync
- Cross-platform (Mac, Windows, Linux)
- Free and open source

**Setup:**
```bash
# Install Syncthing on both machines
brew install syncthing  # macOS

# Start Syncthing
syncthing

# Open web GUI: http://localhost:8384
# Add folder: datasets/
# Share with colleague's Device ID
```

**Transfer Speed:** 
- LAN: 100-1000 Mbps (full gigabit)
- Internet: Limited by upload speed

**Time:**
- Same network: 5-10 minutes
- Internet: 1-3 hours

---

#### **C. rsync over SSH (For Tech-Savvy Colleagues)**
If colleague has SSH server or shared Unix system:

```bash
# One-time transfer
rsync -avP --compress datasets/ colleague@server.edu:/path/to/datasets/

# Or create tarball and transfer
tar -czf - datasets/ | ssh colleague@server.edu "cat > datasets.tar.gz"
```

---

### **Option 4: Torrenting (BEST for Multiple Recipients) ⭐**

#### **Academic BitTorrent / WebTorrent**
**Use Case:** Sharing with 5+ colleagues simultaneously

**Pros:**
- Distributed load (faster for many people)
- Resume support
- Each downloader becomes uploader
- No single server bottleneck

**How to Create:**
```bash
# Install transmission-cli
brew install transmission-cli

# Create torrent
transmission-create -o datasets.torrent \
    -t udp://tracker.opentrackr.org:1337 \
    -c "RT-MonoDepth Benchmark Datasets" \
    datasets/

# Share .torrent file (tiny, ~100KB)
# Seed from your machine
transmission-daemon --download-dir . --seed datasets.torrent
```

**Cons:**
- Requires technical knowledge
- Need to keep seeding
- University firewalls may block

---

## 🎯 **BEST PRACTICES & RECOMMENDATIONS**

### **For Your Specific Case (31 GB):**

#### **Recommended Workflow:**

**1. Immediate Team (1-3 people nearby):**
```bash
# Use External SSD
# Cost: $150 (one-time, reusable)
# Time: 2 minutes per person
# Best: Instant, offline, reliable
```

**2. Remote Colleagues (Research Collaborators):**
```bash
# Use Zenodo (All files fit in single upload!)
# Cost: Free
# Time: 2-3 hours upload (one-time)
# Best: Citable DOI, permanent, public good
```

**3. Lab/Institution Team (5+ people):**
```bash
# Use Institutional NAS/File Server
# Cost: Already available
# Time: 15 minutes initial copy
# Best: Fast LAN transfers, central access
```

---

## 📦 **COMPRESSION OPTIMIZATION**

### **Compress Before Sharing:**

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction

# Option 1: Standard gzip compression (balanced)
tar -czf datasets_all.tar.gz datasets/
# Expected size: ~28 GB (10% savings)
# Time: ~15 minutes

# Option 2: High compression (xz/lzma) - BEST compression
tar -cJf datasets_all.tar.xz datasets/
# Expected size: ~25 GB (19% savings)
# Time: ~1 hour (CPU intensive)

# Option 3: Parallel compression (faster) - RECOMMENDED ⭐
brew install pigz
tar -I pigz -cf datasets_all.tar.gz datasets/
# Time: ~4 minutes (uses all CPU cores)

# Option 4: Split by dataset (recommended for organization)
tar -czf datasets_nyu.tar.gz datasets/nyu_depth_v2/           # 2.5 GB
tar -czf datasets_kitti.tar.gz datasets/kitti/                # 12 GB
tar -czf datasets_cityscapes.tar.gz datasets/cityscapes/      # 13 GB
tar -czf datasets_make3d.tar.gz datasets/make3d/              # 600 MB
```

### **Verify Integrity After Transfer:**

```bash
# Create checksums before sharing
shasum -a 256 datasets_*.tar.gz > datasets_checksums.txt

# Colleague verifies after download
shasum -a 256 -c datasets_checksums.txt
```

---

## 🔒 **SECURITY CONSIDERATIONS**

### **For Public Sharing (Zenodo, figshare):**
✅ **Safe to share publicly:**
- These are standard academic datasets
- Already publicly available
- No proprietary data
- Redistribution allowed by original licenses

⚠️ **Check Original Licenses:**
- **NYU Depth V2:** Academic use allowed ✅
- **KITTI:** Non-commercial use allowed ✅
- **Cityscapes:** Academic use with registration ⚠️ (may need permission)
- **Make3D:** Academic use allowed ✅

**Recommendation for Cityscapes:**
- Don't upload Cityscapes directly to public repos
- Provide download script instead
- Or use private sharing (Google Drive with limited access)

### **For Private Sharing (Google Drive, Institution):**
- All datasets OK to share
- Add README citing original sources
- Include license information

---

## 📝 **SHARING PACKAGE TEMPLATE**

When sharing, include this `README.md` with the datasets:

```markdown
# RT-MonoDepth Construction Site Benchmark Datasets
**Version:** 1.0
**Date:** November 2025
**Size:** 31 GB (uncompressed)

## Contents
- NYU Depth V2 (2.8 GB) - Indoor depth baseline
- KITTI (13 GB) - Outdoor/driving scenes
- Cityscapes (14 GB) - Urban pedestrian scenarios
- Make3D (700 MB) - Diverse outdoor scenes

## Usage
See DATASET_DOWNLOAD_GUIDE.md for structure and evaluation details.

## Original Sources
- NYU: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Cityscapes: https://www.cityscapes-dataset.com/
- Make3D: http://make3d.cs.cornell.edu/

## License
These datasets retain their original licenses. Academic use only.

## Citation
If you use this collection, please cite the original dataset papers:
[Include BibTeX entries]

## Contact
[Your Name] - [Email]
[Institution]
```

---

## ⚡ **QUICK COMPARISON TABLE**

| Method | Cost | Speed | Best For | Difficulty |
|--------|------|-------|----------|------------|
| **External SSD** | $150 | ⭐⭐⭐⭐⭐ (2 min) | Local team | Easy |
| **Google Drive** | Free | ⭐⭐⭐ (1.5 hrs) | Remote individuals | Easy |
| **Zenodo** | Free | ⭐⭐⭐ (2 hrs) | Publication | Easy ⭐ |
| **Institution NAS** | Free | ⭐⭐⭐⭐⭐ (5 min) | Lab team | Easy |
| **Syncthing P2P** | Free | ⭐⭐⭐⭐ (15 min) | Tech-savvy peers | Medium |
| **Torrenting** | Free | ⭐⭐⭐⭐ (varies) | Many recipients | Hard |

---

## 🎯 **MY RECOMMENDATION FOR YOU:**

Based on your Q1 journal publication timeline:

### **Primary Method: Zenodo (Single Upload!) ⭐**
1. ✅ Upload all datasets at once (31GB total, under 50GB limit!)
2. ✅ Get DOI for the dataset collection
3. ✅ Include in paper supplementary materials
4. ✅ Cite in your methodology section
5. ✅ Reviewers can access easily
6. ✅ Permanent archival (10+ years)

### **Backup Method: Institutional Storage**
- For quick sharing with lab members
- Fast LAN transfers
- Private until publication

### **On-Demand: Google Drive**
- For reviewers or collaborators who need quick access
- Create compressed archives
- Share with specific permissions

---

## 🚀 **ACTION PLAN**

```bash
# Step 1: Create compressed archives (for uploading)
cd /Users/ashwani/Desktop/RT-Monodepth-Construction

# Parallel compression (fastest)
brew install pigz
tar -I pigz -cf datasets_nyu_make3d.tar.gz datasets/nyu_depth_v2/ datasets/make3d/
tar -I pigz -cf datasets_kitti.tar.gz datasets/kitti/
tar -I pigz -cf datasets_cityscapes.tar.gz datasets/cityscapes/

# Step 2: Generate checksums
shasum -a 256 datasets_*.tar.gz > CHECKSUMS.txt

# Step 3: Create README
cat > DATASET_README.md << 'EOF'
# RT-MonoDepth Benchmark Datasets
[Include the template above]
EOF

# Step 4: Upload to Zenodo
# (Manual: https://zenodo.org/deposit/new)

# Step 5: Share links with team
```

---

## 📊 **ESTIMATED COSTS**

**Free Options:**
- Zenodo: Free (unlimited)
- Institutional storage: Free (already paid)
- Google Drive: Free (edu account unlimited)
- Syncthing: Free

**Paid Options:**
- External SSD: $150-200 (one-time, reusable)
- Cloud storage (commercial): $10-20/month

**Recommended Investment:** $0 (use Zenodo + institutional storage)

---

## ⏱️ **TIME INVESTMENT**

**One-Time Setup:**
- Compression: 4-5 minutes (parallel with pigz)
- Upload to Zenodo: 2-3 hours (background)
- Documentation: 30 minutes
- **Total: 3-4 hours (mostly automated)**

**Per Colleague:**
- Send link: 1 minute
- Their download: Automatic

**ROI:** After 2-3 colleagues, Zenodo pays for itself vs. individual transfers!

---

## 🆘 **TROUBLESHOOTING**

**Q: Upload keeps failing?**
- Split into smaller chunks (10GB each)
- Use resume-capable tools (rclone, mega-cmd)

**Q: Colleague can't download?**
- Check firewall/proxy settings
- Use alternative method (Drive vs. Zenodo)
- Try torrent/P2P

**Q: Too slow?**
- Check upload bandwidth (test at speedtest.net)
- Use compression (-j for parallel)
- Consider local transfer (USB drive)

**Q: Storage limit reached?**
- Clean up duplicates
- Use institutional unlimited storage
- Split across multiple free accounts

---

## 📚 **ADDITIONAL RESOURCES**

**Tools:**
- rclone: https://rclone.org/ (cloud transfer automation)
- Syncthing: https://syncthing.net/ (P2P sync)
- transmission: https://transmissionbt.com/ (torrenting)
- pigz: https://github.com/madler/pigz (parallel gzip)

**Services:**
- Zenodo: https://zenodo.org/
- figshare: https://figshare.com/
- IEEE DataPort: https://ieee-dataport.org/
- OSF: https://osf.io/ (Open Science Framework)

---

**Questions?** Open an issue or contact the team.

**Next Steps:** Choose your sharing method and start uploading! 🚀
