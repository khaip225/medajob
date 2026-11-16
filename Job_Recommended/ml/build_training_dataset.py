"""
Build the training dataset for the job-recommendation GNN (Kaggle-ready).

Sources = the data the backend actually serves (same rows as backend/app/job_text_emb.npy
(39,036) and backend/app/user_text_emb.npy (3,983)):
  - import_db/COMBINED_DATA_PROCESSED1.csv   (jobs)
  - import_db/USER_DATA_PROCESSED1.csv       (job seekers)
If the CSVs are no longer in import_db/, they are extracted from git history automatically.

What it fixes compared with feeding the raw files to the notebook
  * jobs  : duplicate postings removed; 'Khác' province recovered from the street address
            (districts / industrial parks / English spellings); 'Khác' industry recovered from
            title + main_industry; salary_type 3 ("thỏa thuận", numbers were imputed) -> NaN;
            "Trên X"/"Đến X" salaries no longer carry the imputed bound; minimum degree parsed
            from the requirement text; job level from job_function/title; benefits -> 9 groups.
  * users : duplicate UserIDs and cloned profiles removed; desired salary parsed from the
            original string (the numeric columns were mis-scaled, e.g. -0.1667); degree level
            parsed from the education text; multi-province wishes kept; occupation + level +
            part-time intent extracted; templated driver profiles (50% of rows) capped.
  * both  : one shared occupation taxonomy (35 roles) and the 14 industry groups;
            provinces mapped to the 34 post-2025 units (NQ 202/2025/QH15).
  * labels: no real interaction log exists, so applications are SIMULATED with a documented
            behaviour model (see simulate_applications) that mixes content fit, job
            attractiveness, a latent collaborative component and constraint penalties whose
            strength depends on observable user attributes. Each user's positives are split
            70/15/15 into train/val/test. Hard negatives = relevant-but-infeasible jobs.

Usage (from Job_Recommended/):
    python -m ml.build_training_dataset                  # -> ../kaggle_dataset
    python -m ml.build_training_dataset --out D:/x --driver-cap 500 --check
"""

import argparse
import json
import math
import os
import re
import subprocess
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # Job_Recommended/
IMPORT_DB = os.path.join(ROOT, "import_db")
GIT_SOURCES = {  # file -> commit that still contains it
    "COMBINED_DATA_PROCESSED1.csv": "3067889",
    "USER_DATA_PROCESSED1.csv": "bdd4ad1",
}
SEED = 2026


# =====================================================================================
# 0. Loading
# =====================================================================================
def load_source(name):
    p = os.path.join(IMPORT_DB, name)
    if os.path.exists(p):
        print(f"[load] {p}")
        return pd.read_csv(p, low_memory=False)
    commit = GIT_SOURCES[name]
    print(f"[load] {name} not in import_db/ -> git show {commit}:import_db/{name}")
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:import_db/{name}"],
                         capture_output=True, check=True).stdout
    return pd.read_csv(StringIO(raw.decode("utf-8-sig", errors="replace")), low_memory=False)


# =====================================================================================
# 1. Text helpers
# =====================================================================================
def fold(s):
    """Lowercase, strip Vietnamese diacritics, collapse whitespace."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", s).strip()


def clean_text(s, n=None):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:n] if n else s


# =====================================================================================
# 2. Locations: 63 old provinces -> 34 new units (NQ 202/2025/QH15)
# =====================================================================================
UNITS = {  # new unit: (lat, lon, [old provinces merged into it])
    "Hà Nội": (21.03, 105.85, ["Hà Nội"]), "Huế": (16.46, 107.59, ["Thừa Thiên Huế", "Huế"]),
    "Lai Châu": (22.40, 103.45, ["Lai Châu"]), "Điện Biên": (21.39, 103.02, ["Điện Biên"]),
    "Sơn La": (21.33, 103.91, ["Sơn La"]), "Lạng Sơn": (21.85, 106.76, ["Lạng Sơn"]),
    "Quảng Ninh": (21.01, 107.29, ["Quảng Ninh"]), "Thanh Hóa": (19.81, 105.78, ["Thanh Hóa"]),
    "Nghệ An": (18.9, 105.3, ["Nghệ An"]), "Hà Tĩnh": (18.34, 105.91, ["Hà Tĩnh"]),
    "Cao Bằng": (22.67, 106.26, ["Cao Bằng"]),
    "Tuyên Quang": (22.3, 105.1, ["Tuyên Quang", "Hà Giang"]), "Lào Cai": (22.1, 104.2, ["Lào Cai", "Yên Bái"]),
    "Thái Nguyên": (21.8, 105.9, ["Thái Nguyên", "Bắc Kạn"]),
    "Phú Thọ": (21.3, 105.2, ["Phú Thọ", "Vĩnh Phúc", "Hòa Bình"]),
    "Bắc Ninh": (21.2, 106.1, ["Bắc Ninh", "Bắc Giang"]), "Hưng Yên": (20.7, 106.2, ["Hưng Yên", "Thái Bình"]),
    "Hải Phòng": (20.85, 106.68, ["Hải Phòng", "Hải Dương"]),
    "Ninh Bình": (20.3, 105.98, ["Ninh Bình", "Hà Nam", "Nam Định"]),
    "Quảng Trị": (17.2, 106.8, ["Quảng Trị", "Quảng Bình"]), "Đà Nẵng": (15.9, 108.2, ["Đà Nẵng", "Quảng Nam"]),
    "Quảng Ngãi": (15.0, 108.4, ["Quảng Ngãi", "Kon Tum"]), "Gia Lai": (14.0, 108.5, ["Gia Lai", "Bình Định"]),
    "Khánh Hòa": (12.2, 109.1, ["Khánh Hòa", "Ninh Thuận"]),
    "Lâm Đồng": (11.7, 108.2, ["Lâm Đồng", "Đắk Nông", "Bình Thuận"]),
    "Đắk Lắk": (13.0, 108.3, ["Đắk Lắk", "Phú Yên"]),
    "TP. Hồ Chí Minh": (10.8, 106.7, ["Hồ Chí Minh", "Bình Dương", "Bà Rịa - Vũng Tàu"]),
    "Đồng Nai": (11.1, 107.0, ["Đồng Nai", "Bình Phước"]), "Tây Ninh": (11.2, 106.2, ["Tây Ninh", "Long An"]),
    "Cần Thơ": (9.9, 105.8, ["Cần Thơ", "Sóc Trăng", "Hậu Giang"]),
    "Vĩnh Long": (10.1, 106.0, ["Vĩnh Long", "Bến Tre", "Trà Vinh"]),
    "Đồng Tháp": (10.4, 105.9, ["Đồng Tháp", "Tiền Giang"]), "Cà Mau": (9.2, 105.2, ["Cà Mau", "Bạc Liêu"]),
    "An Giang": (10.2, 105.1, ["An Giang", "Kiên Giang"]),
}
UNIT_NAMES = list(UNITS)
UNIT_IDX = {n: i for i, n in enumerate(UNIT_NAMES)}
ANY = "Toàn quốc"

_PROV2UNIT = {}
for _new, (_, _, _olds) in UNITS.items():
    _PROV2UNIT[fold(_new)] = _new
    for _o in _olds:
        _PROV2UNIT[fold(_o)] = _new
_PROV2UNIT.update({
    "ba ria vung tau": "TP. Hồ Chí Minh", "ba ria-vung tau": "TP. Hồ Chí Minh", "brvt": "TP. Hồ Chí Minh",
    "ho chi minh": "TP. Hồ Chí Minh", "tp ho chi minh": "TP. Hồ Chí Minh", "tp. ho chi minh": "TP. Hồ Chí Minh",
    "hcm": "TP. Hồ Chí Minh", "tphcm": "TP. Hồ Chí Minh", "tp hcm": "TP. Hồ Chí Minh", "tp.hcm": "TP. Hồ Chí Minh",
    "sai gon": "TP. Hồ Chí Minh", "saigon": "TP. Hồ Chí Minh", "ho chi minh city": "TP. Hồ Chí Minh",
    "hanoi": "Hà Nội", "ha noi city": "Hà Nội", "thua thien hue": "Huế", "hue": "Huế",
    "dak lak": "Đắk Lắk", "daklak": "Đắk Lắk", "dac lak": "Đắk Lắk", "dak nong": "Lâm Đồng", "dac nong": "Lâm Đồng",
    "hai phong city": "Hải Phòng", "da nang city": "Đà Nẵng", "binh duong": "TP. Hồ Chí Minh",
})

# District / industrial park / landmark gazetteer (folded) used only when the province field is unknown.
_GAZ = {
    "TP. Hồ Chí Minh": """thu duc|binh thanh|tan binh|tan phu|go vap|phu nhuan|binh tan|hoc mon|cu chi|nha be|binh chanh|
        can gio|district [0-9]{1,2}|tan thuan|linh trung|vinh loc|tan tao|quang trung software|qtsc|thao dien|an phu|
        an khanh|thu thiem|thanh my loi|phu my hung|hiep phuoc|cat lai|tan son nhat|nhieu loc|xuan hoa|binh trung|
        thanh my tay|tan thong hoi|cong hoa|truong chinh|song than|vsip|vietnam.?singapore|di an|thuan an|
        thu dau mot|ben cat|tan uyen|bau bang|my phuoc|dau tieng|phu giao|vung tau|ba ria|phu my|chau duc|dat do|
        long dien|xuyen moc|nguyen hue|ben thanh|landmark 81|bitexco|etown|pico plaza|sunshine center|q\\.? ?(?:1[0-2]|[1-9])\\b|
        quan (?:1[0-2]|[1-9])\\b""",
    "Hà Nội": """ba dinh|hoan kiem|hai ba trung|dong da|cau giay|thanh xuan|hoang mai|long bien|tay ho|ha dong|tu liem|
        my dinh|gia lam|dong anh|soc son|me linh|hoai duc|thach that|quoc oai|chuong my|thanh tri|thuong tin|
        dan phuong|phuc tho|son tay|ba vi|thang long industrial|noi bai|pham hung|khuat duy tien|lieu giai|duy tan|
        trung hoa|yen hoa|dich vong|phuc dien|trung tu|ton that tung|ham nghi|keangnam|lotte center|times city|
        royal city|nguyen chi thanh|xuan thuy|tran duy hung|giai phong|lang ha|thai ha|kim ma|van phuc""",
    "Đồng Nai": "bien hoa|long thanh|nhon trach|trang bom|amata|loteco|long khanh|dau giay|bau xeo|dong xoai|chon thanh|binh phuoc",
    "Hải Phòng": "dinh vu|cat hai|trang due|an duong|thuy nguyen|do son|nam cau kien|deep c|hai duong|chi linh|nam sach|cam giang",
    "Đà Nẵng": "hai chau|thanh khe|son tra|ngu hanh son|lien chieu|cam le|hoa khanh|hoi an|tam ky|dien ban|dien nam",
    "Bắc Ninh": "tu son|yen phong|que vo|tien du|thuan thanh|viet yen|quang chau|dinh tram|van trung|song khe|bac giang",
    "Hưng Yên": "my hao|van lam|pho noi|van giang|thai binh",
    "Phú Thọ": "viet tri|vinh yen|phuc yen|binh xuyen|ba thien|vinh phuc|hoa binh",
    "Quảng Ninh": "ha long|cam pha|mong cai|uong bi|quang yen|dong trieu",
    "Khánh Hòa": "nha trang|cam ranh|ninh thuan|phan rang",
    "Lâm Đồng": "da lat|bao loc|phan thiet|binh thuan|dak nong|gia nghia",
    "Nghệ An": "tp vinh|thanh pho vinh|vsip nghe an|cua lo",
    "Huế": "phu bai",
    "Gia Lai": "quy nhon|binh dinh|pleiku|nhon hoi",
    "Đắk Lắk": "buon ma thuot|phu yen|tuy hoa",
    "Tây Ninh": "tan an|duc hoa|ben luc|can giuoc|can duoc|long an|trang bang|phuoc dong",
    "Đồng Tháp": "my tho|tien giang|cao lanh|sa dec",
    "An Giang": "rach gia|phu quoc|long xuyen|chau doc|kien giang",
    "Cần Thơ": "ninh kieu|cai rang|binh thuy|tra noc|soc trang|vi thanh|hau giang",
    "Ninh Bình": "phu ly|nam dinh|ha nam|tam diep|dong van",
    "Thái Nguyên": "song cong|pho yen|yen binh|bac kan",
    "Thanh Hóa": "nghi son|sam son|bim son",
    "Hà Tĩnh": "vung ang|ky anh",
    "Quảng Ngãi": "dung quat|kon tum",
    "Quảng Trị": "dong hoi|quang binh|dong ha",
    "Vĩnh Long": "ben tre|tra vinh",
    "Cà Mau": "bac lieu",
}
_PROV_RE = [(re.compile(r"\b" + re.escape(k) + r"\b"), v) for k, v in sorted(_PROV2UNIT.items(), key=lambda x: -len(x[0]))]
_GAZ_RE = [(re.compile(r"\b(?:" + re.sub(r"\s*\n\s*", "", pat) + r")"), unit) for unit, pat in _GAZ.items()]
_ANY_RE = re.compile(r"toan quoc|ca nuoc|nhieu (noi|tinh)|remote|lam viec tu xa|work from home|tat ca tinh")


def units_from_text(s, use_gazetteer=True):
    """Free text -> sorted list of unit names (or [ANY]). Province names win over the gazetteer."""
    f = fold(s)
    if not f:
        return []
    if _ANY_RE.search(f):
        return [ANY]
    found = []
    for rx, unit in _PROV_RE:
        if rx.search(f):
            found.append(unit)
    if not found and use_gazetteer:
        for rx, unit in _GAZ_RE:
            if rx.search(f):
                found.append(unit)
                break  # one address -> one unit
    return sorted(set(found), key=lambda u: UNIT_IDX[u])


def units_from_list(s):
    """'Hồ Chí Minh, Bình Dương' -> units (no gazetteer, list semantics)."""
    out = []
    for tok in re.split(r"[,;/|]| va ", fold(s)):
        tok = tok.strip()
        if not tok:
            continue
        if _ANY_RE.search(tok):
            return [ANY]
        u = _PROV2UNIT.get(tok)
        if u is None:
            us = units_from_text(tok, use_gazetteer=False)
            out += us
        else:
            out.append(u)
    return sorted(set(x for x in out if x != ANY), key=lambda u: UNIT_IDX[u])


def haversine_matrix():
    lat = np.radians([UNITS[n][0] for n in UNIT_NAMES])
    lon = np.radians([UNITS[n][1] for n in UNIT_NAMES])
    a = np.sin((lat[:, None] - lat[None]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2
    return (2 * 6371 * np.arcsin(np.sqrt(a))).astype(np.float32)


# =====================================================================================
# 3. Occupation taxonomy (shared by jobs and users) + industry group + level
# =====================================================================================
# (code, display name, industry_group, regex on folded text). ORDER MATTERS: first match wins.
OCCUPATIONS = [
    ("lai_xe",        "Lái xe / Tài xế",                    "Vận tải - Logistics",      r"lai xe|tai xe|\bdriver\b|phu xe|lo xe|xe nang|xe tai|xe container|xe khach"),
    ("it_dev",        "Lập trình / Phát triển phần mềm",    "Công nghệ thông tin",      r"developer|lap trinh|software|backend|back-end|frontend|front-end|fullstack|full stack|\bjava\b|python|\bphp\b|\.net|reactjs|react native|nodejs|golang|\bios\b|android|flutter|devops|\bai engineer|machine learning|deep learning|data engineer|data scien|blockchain|game dev|unity|embedded|firmware|it phan mem|cong nghe phan mem|solution architect"),
    ("it_test",       "Kiểm thử phần mềm",                  "Công nghệ thông tin",      r"tester|kiem thu phan mem|kiem thu(?! ngan)|automation test|manual test|\bqa engineer|qa/qc phan mem|\bsqa\b"),
    ("it_infra",      "IT hạ tầng / Hỗ trợ kỹ thuật",       "Công nghệ thông tin",      r"it support|it helpdesk|helpdesk|it phan cung|phan cung|quan tri mang|network|system admin|sysadmin|quan tri he thong|cloud|security engineer|an ninh mang|bao mat|\bit\b|cong nghe thong tin|ky thuat vien it"),
    ("data_ba",       "Phân tích dữ liệu / BA",             "Công nghệ thông tin",      r"data analyst|phan tich du lieu|business analyst|\bba\b|\bbi\b|power bi|product owner|product manager|\bpo\b"),
    ("ke_toan",       "Kế toán / Kiểm toán",                "Tài chính - Kế toán",      r"ke toan|kiem toan|accountant|accounting|auditor|\baudit|\bthue\b|\btax\b|thu ngan|cashier|thu quy"),
    ("tai_chinh",     "Ngân hàng / Tài chính / Bảo hiểm",   "Tài chính - Kế toán",      r"ngan hang|tin dung|giao dich vien|teller|banking|\bbank|tai chinh|finance|financial|dau tu|chung khoan|bao hiem|insurance|tham dinh|thu hoi no|credit|quan ly rui ro|\brisk\b"),
    ("xnk",           "Xuất nhập khẩu / Chứng từ",          "Vận tải - Logistics",      r"xuat nhap khau|xuat khau|nhap khau|\bxnk\b|import|export|chung tu|customs|hai quan|forwarder|forwarding|freight|\bcus\b"),
    ("thu_mua",       "Thu mua / Vật tư / Cung ứng",        "Vận tải - Logistics",      r"thu mua|mua hang|purchas|procurement|vat tu|cung ung|supply chain|chuoi cung ung|\bsourcing"),
    ("kho_van",       "Kho vận / Giao nhận",                "Vận tải - Logistics",      r"\bkho\b|thu kho|warehouse|logistic|giao nhan|giao hang|shipper|delivery|buu ta|buu chinh|dieu phoi van tai|van tai|dieu do|boc xep|dong goi"),
    ("marketing",     "Marketing / Truyền thông",           "Marketing - Truyền thông", r"marketing|content|\bseo\b|digital|truyen thong|\bpr\b|brand|thuong hieu|social media|quang cao|copywriter|livestream|tiktok|bien tap|nha bao|phong vien|\bmc\b|su kien|event|kol|ecommerce|thuong mai dien tu|san thuong mai"),
    ("thiet_ke",      "Thiết kế / Sáng tạo",                "Marketing - Truyền thông", r"thiet ke do hoa|graphic|designer|ui/ux|\bux\b|\bui\b|my thuat|video editor|dung phim|edit video|quay phim|nhiep anh|photograph|\b2d\b|\b3d\b|animation|hoa si|art director|thiet ke thoi trang|thiet ke$|nhan vien thiet ke"),
    ("cskh",          "Chăm sóc khách hàng / Telesales",    "Kinh doanh - Bán hàng",    r"cham soc khach hang|\bcskh\b|customer service|customer care|customer support|telesale|tong dai|call center|dich vu khach hang|tu van vien|tu van khach hang|hau mai"),
    ("bds",           "Bất động sản / Môi giới",            "Xây dựng - BĐS",           r"bat dong san|\bbds\b|moi gioi|real estate|du an bat dong san"),
    ("kinh_doanh",    "Kinh doanh / Bán hàng",              "Kinh doanh - Bán hàng",    r"kinh doanh|\bsales?\b|ban hang|business development|account manager|key account|phat trien thi truong|thi truong|dai dien thuong mai|cua hang truong|store manager|merchandis|\bkd\b|\bbd\b|tu van ban|showroom|sieu thi|ban le|retail"),
    ("nhan_su",       "Nhân sự / Tuyển dụng",               "Hành chính - Nhân sự",     r"nhan su|\bhr\b|hrbp|tuyen dung|recruit|talent|c&b|payroll|tien luong|dao tao noi bo|human resource"),
    ("phap_che",      "Pháp chế / Luật",                    "Hành chính - Nhân sự",     r"phap che|phap ly|\bluat\b|legal|lawyer|compliance|tuan thu|cong chung"),
    ("phien_dich",    "Biên phiên dịch / Ngoại ngữ",        "Hành chính - Nhân sự",     r"phien dich|bien dich|interpreter|translator|tieng trung|tieng nhat|tieng han|tieng duc|tieng phap|tieng thai|tieng dai|ngon ngu"),
    ("hanh_chinh",    "Hành chính / Văn phòng / Trợ lý",    "Hành chính - Nhân sự",     r"hanh chinh|van phong|\badmin\b|le tan|receptionist|thu ky|tro ly|assistant|secretary|nhap lieu|data entry|danh may|van thu|office"),
    ("giao_duc",      "Giáo dục / Đào tạo",                 "Giáo dục - Đào tạo",       r"giao vien|giang vien|tro giang|teacher|tutor|gia su|dao tao|giao duc|education|trainer|coach|mam non|tieng anh"),
    ("y_te",          "Y tế / Dược",                        "Y tế - Dược",              r"dieu duong|bac si|y si|y ta|duoc si|duoc|nurse|doctor|pharma|xet nghiem|ky thuat vien y|nha khoa|nha si|dental|phong kham|benh vien|y te|ho ly|thu y|trinh duoc vien"),
    ("xay_dung",      "Xây dựng / Kiến trúc",               "Xây dựng - BĐS",           r"xay dung|ket cau|kien truc|giam sat cong trinh|cong trinh|du toan|\bqs\b|\bm&e\b|\bmep\b|civil|site engineer|noi that|interior|architect|trac dia|cau duong|thi cong|ha tang|shop ?drawing|autocad|revit"),
    ("dien",          "Điện / Điện tử / Tự động hóa",       "Kỹ thuật - Sản xuất",      r"\bdien\b(?! thoai)|dien tu|dien lanh|electrical|electric|electronic|\bplc\b|tu dong hoa|automation|vien thong|telecom|smt|\bpcb\b|nang luong|solar"),
    ("co_khi",        "Cơ khí / Chế tạo",                   "Kỹ thuật - Sản xuất",      r"co khi|mechanical|che tao|\bcnc\b|\bthe han\b|\bhan\b|tien phay|khuon|o to|oto|automotive|xe may|dong co|may moc|co dien"),
    ("bao_tri",       "Bảo trì / Kỹ thuật viên",            "Kỹ thuật - Sản xuất",      r"bao tri|bao duong|maintenance|sua chua|ky thuat vien|technician|ky thuat|ky su|engineer"),
    ("qa_qc",         "QA / QC sản xuất",                   "Kỹ thuật - Sản xuất",      r"\bqc\b|\bqa\b|kiem tra chat luong|quality|\bkcs\b|\biso\b|chat luong|kiem dinh|\bhse\b|an toan lao dong|\behs\b"),
    ("san_xuat",      "Sản xuất / Vận hành",                "Kỹ thuật - Sản xuất",      r"san xuat|production|cong nhan|van hanh|operator|xuong|to truong|planner|ke hoach san xuat|\bppc\b|det may|may mac|cong nhan may|tho may|in an|bao bi|thuc pham|hoa chat|hoa hoc|sinh hoc|moi truong|xu ly nuoc|lab|phong thi nghiem|r&d"),
    ("fnb",           "Nhà hàng / Khách sạn / F&B",         "Dịch vụ - F&B - Làm đẹp",  r"phuc vu|\bbep\b|dau bep|chef|bartender|pha che|barista|nha hang|khach san|buong phong|f&b|restaurant|hotel|\bcook|tap vu nha hang|le tan khach san|cafe|tra sua"),
    ("lam_dep",       "Làm đẹp / Spa / Thể hình",           "Dịch vụ - F&B - Làm đẹp",  r"\bspa\b|\bnail\b|tham my|lam dep|massage|stylist|cat toc|\btoc\b|my pham|\bpt\b|personal trainer|yoga|gym|the hinh|beauty"),
    ("du_lich",       "Du lịch / Lữ hành",                  "Dịch vụ - F&B - Làm đẹp",  r"du lich|\btour\b|huong dan vien|dieu hanh tour|ve may bay|booking|travel|hang khong|tiep vien"),
    ("nong_nghiep",   "Nông - Lâm - Ngư nghiệp",            "Nông - Lâm - Ngư nghiệp",  r"nong nghiep|chan nuoi|thuy san|trong trot|lam nghiep|nong san|cay trong|ky su nong|phan bon|thuc an chan nuoi"),
    ("lao_dong_pt",   "Lao động phổ thông / Bảo vệ / Tạp vụ", "Lao động phổ thông",     r"lao dong pho thong|tap vu|bao ve|security guard|giup viec|ve sinh|phu kho|lam them|thoi vu|part ?time|ban thoi gian|viec lam online|tai nha|cong tac vien|\bctv\b|freelancer|nhan vien$"),
]
OCC_CODES = [o[0] for o in OCCUPATIONS] + ["khac"]
OCC_IDX = {c: i for i, c in enumerate(OCC_CODES)}
OCC_NAME = {o[0]: o[1] for o in OCCUPATIONS} | {"khac": "Khác"}
OCC_GROUP = {o[0]: o[2] for o in OCCUPATIONS} | {"khac": "Khác"}
_OCC_RE = [(c, re.compile(p)) for c, _, _, p in OCCUPATIONS]

INDUSTRY_GROUPS = ["Kinh doanh - Bán hàng", "Tài chính - Kế toán", "Kỹ thuật - Sản xuất", "Hành chính - Nhân sự",
                   "Xây dựng - BĐS", "Marketing - Truyền thông", "Công nghệ thông tin", "Dịch vụ - F&B - Làm đẹp",
                   "Vận tải - Logistics", "Giáo dục - Đào tạo", "Y tế - Dược", "Lao động phổ thông",
                   "Nông - Lâm - Ngư nghiệp", "Khác"]

# Related occupations (symmetric similarity, diagonal = 1). Used by the behaviour model only.
OCC_RELATED = [
    (["lai_xe", "kho_van"], .45), (["kho_van", "xnk", "thu_mua"], .45), (["lai_xe", "lao_dong_pt"], .25),
    (["it_dev", "it_test", "it_infra", "data_ba"], .45), (["ke_toan", "tai_chinh"], .45), (["ke_toan", "hanh_chinh"], .2),
    (["kinh_doanh", "cskh", "bds", "marketing"], .35), (["kinh_doanh", "cskh"], .55), (["marketing", "thiet_ke"], .4),
    (["hanh_chinh", "nhan_su", "phap_che", "phien_dich"], .4), (["hanh_chinh", "cskh"], .25),
    (["co_khi", "dien", "bao_tri", "qa_qc", "san_xuat"], .35), (["dien", "bao_tri"], .55), (["co_khi", "bao_tri"], .55),
    (["qa_qc", "san_xuat"], .55), (["xay_dung", "dien", "co_khi"], .2), (["xay_dung", "bds"], .2),
    (["fnb", "du_lich", "lam_dep"], .3), (["san_xuat", "lao_dong_pt"], .35), (["fnb", "lao_dong_pt"], .25),
    (["phien_dich", "giao_duc"], .35), (["y_te", "lam_dep"], .2), (["nong_nghiep", "san_xuat"], .25),
]


def occ_sim_matrix():
    O = len(OCC_CODES)
    S = np.eye(O, dtype=np.float32)
    for group, w in OCC_RELATED:
        for a in group:
            for b in group:
                if a != b:
                    S[OCC_IDX[a], OCC_IDX[b]] = max(S[OCC_IDX[a], OCC_IDX[b]], w)
    S[OCC_IDX["khac"], OCC_IDX["khac"]] = 0.3   # 'unknown' matches 'unknown' only weakly
    return S


def occupations_of(*texts, max_n=2):
    """First matching occupations scanning texts in priority order (e.g. title, then main_industry)."""
    found = []
    for t in texts:
        f = fold(t)
        if not f:
            continue
        # earlier taxonomy entries are more specific -> keep taxonomy order
        for c, rx in _OCC_RE:
            if c not in found and rx.search(f):
                found.append(c)
            if len(found) >= max_n:
                break
        if found:
            break
    return found or ["khac"]


LEVELS = {0: "Thực tập / Sinh viên", 1: "Nhân viên", 2: "Trưởng nhóm / Giám sát", 3: "Trưởng phòng / Quản lý", 4: "Giám đốc"}
_FUNC_LEVEL = {
    "Thực tập sinh": 0, "Sinh viên/ Thực tập sinh": 0, "Mới tốt nghiệp": 1, "Nhân viên": 1,
    "Trưởng nhóm": 2, "Trưởng nhóm / Giám sát": 2, "Tổ trưởng": 2, "Phó tổ trưởng": 2, "Quản lý / Giám sát": 2,
    "Quản lý": 3, "Trưởng phòng": 3, "Phó trưởng phòng": 3, "Trưởng/Phó phòng": 3, "Quản lý cấp trung": 3,
    "Trưởng chi nhánh": 3, "Quản lý cấp cao": 4, "Giám đốc": 4, "Phó Giám đốc": 4, "Phó giám đốc": 4, "Tổng giám đốc": 4,
}
_LEVEL_RE = [
    (4, re.compile(r"giam doc|director|\bceo\b|\bcfo\b|\bcto\b|\bcoo\b|head of")),
    (3, re.compile(r"truong phong|pho phong|quan ly|manager|truong bo phan|truong chi nhanh|ke toan truong")),
    (2, re.compile(r"truong nhom|giam sat|leader|\blead\b|supervisor|to truong|truong ca|senior")),
    (0, re.compile(r"thuc tap|intern|sinh vien|fresher|moi tot nghiep|part ?time|ban thoi gian|lam them|thoi vu")),
]


def level_from_text(t, default=1):
    f = fold(t)
    for lv, rx in _LEVEL_RE:
        if rx.search(f):
            return lv
    return default


# =====================================================================================
# 4. Degree / salary / experience / benefits
# =====================================================================================
DEGREE_NAMES = {0: "THPT", 1: "Trung cấp / Nghề", 2: "Cao đẳng", 3: "Đại học", 4: "Sau đại học"}
_DEG_PAT = [
    (4, r"thac si|tien si|sau dai hoc|\bmaster|\bphd\b|\bmba\b|cao hoc"),
    (3, r"dai hoc|cu nhan|bachelor|university|\bdh\b|hoc vien"),
    (2, r"cao dang|\bcd\b|college"),
    (1, r"trung cap|so cap|day nghe|hoc nghe|truong nghe|chung chi nghe|trung hoc chuyen nghiep|\btcn\b"),
    (0, r"\bthpt\b|pho thong|cap 3|cap ba|lop 12|12/12|9/12|thcs|trung hoc co so|tot nghiep trung hoc"),
]
_DEG_RE = [(lv, re.compile(p)) for lv, p in _DEG_PAT]
_NO_DEG_RE = re.compile(r"khong (yeu cau|can|doi hoi) (bang cap|trinh do|hoc van)|khong yeu cau bang")


def job_min_degree(req):
    """Minimum accepted degree from requirement text; 'ưu tiên X' mentions ignored. NaN if not stated."""
    f = fold(req)
    if not f:
        return np.nan
    if _NO_DEG_RE.search(f):
        return 0.0
    lv = []
    for level, rx in _DEG_RE:
        for m in rx.finditer(f):
            ctx = f[max(0, m.start() - 30):m.start()]
            if "uu tien" in ctx or "loi the" in f[m.end():m.end() + 15]:
                continue
            lv.append(level)
    return float(min(lv)) if lv else np.nan


def user_max_degree(txt):
    f = fold(txt)
    if not f or ("don vi dao tao" not in f and not any(rx.search(f) for _, rx in _DEG_RE)):
        return np.nan
    lv = [level for level, rx in _DEG_RE if rx.search(f)]
    if lv:
        return float(max(lv))
    if re.search(r"\btruong\b|hoc vien|university|college", f):
        return 3.0 if "hoc vien" in f or "university" in f else np.nan
    return np.nan


def job_salary(row):
    """-> (min_trieu, max_trieu, kind). salary_type: 0 range, 1 'Trên X', 2 'Đến X', 3 negotiable (imputed)."""
    t = int(row["salary_type"]) if pd.notna(row["salary_type"]) else 3
    lo, hi = row["salary_min"] / 1e6, row["salary_max"] / 1e6
    if t == 3:
        return np.nan, np.nan, "negotiable"
    if t == 1:
        hi = np.nan
        kind = "from"
    elif t == 2:
        lo = np.nan
        kind = "upto"
    else:
        kind = "range"
    vals = [v for v in (lo, hi) if pd.notna(v)]
    if not vals or max(vals) > 150 or min(vals) < 1:     # outliers / hourly
        return np.nan, np.nan, "missing"
    if pd.notna(lo) and pd.notna(hi) and hi > 5 * lo:    # e.g. "9 - 75 triệu": broken upper bound
        hi, kind = np.nan, "from"
    return lo, hi, kind


def user_salary(s):
    f = fold(s)
    if not f or "thoa thuan" in f:
        return np.nan, np.nan, "negotiable"
    nums = [float(x.replace(",", "").replace(".", "")) / 1e6 for x in re.findall(r"\d[\d,\.]*", f)]
    nums = [n for n in nums if n > 0]
    if not nums:
        return np.nan, np.nan, "negotiable"
    if "tren" in f:
        return nums[0], np.nan, "from"
    return min(nums[:2]), max(nums[:2]), "range"


BEN_CATS = {"Bảo hiểm": r"bao hiem|bhxh|bhyt|insurance", "Thưởng": r"thuong|bonus|thang 13|13th",
            "Phụ cấp": r"phu cap|an trua|com trua|allowance|xang xe|dien thoai|meal|lunch|laptop",
            "Đào tạo/thăng tiến": r"dao tao|training|thang tien|phat trien|career",
            "Khám sức khỏe": r"kham suc khoe|suc khoe|health|medical",
            "Nghỉ phép/du lịch": r"nghi phep|du lich|team ?building|nghi mat|travel|leave|company trip",
            "Linh hoạt/remote": r"remote|linh hoat|tu xa|hybrid|flexible|work from home",
            "Nhà ở/xe đưa đón": r"ky tuc|nha o|xe dua don|xe dua|shuttle|accommodation",
            "Lễ tết/sinh nhật": r"le tet|\btet\b|sinh nhat|birthday|ngay le"}
_BEN_RE = [re.compile(p) for p in BEN_CATS.values()]

PARTTIME_EMP = {"part_time", "freelance", "intern"}
_PT_USER_RE = re.compile(r"part ?time|ban thoi gian|lam them|thoi vu|sinh vien|viec lam online|tai nha|freelanc|cong tac vien")


# =====================================================================================
# 5. Clean jobs
# =====================================================================================
def build_jobs(raw):
    print(f"[jobs] raw rows: {len(raw)}")
    df = raw.copy()
    df["db_job_id"] = np.arange(1, len(df) + 1)        # id assigned by import_jobs.py on a fresh DB
    df = df[df["job_title"].notna() & (df["job_title"].astype(str).str.strip() != "")]

    # ---- location ----
    def _loc(r):
        p = str(r["province"]) if pd.notna(r["province"]) else ""
        if p == "Toàn quốc":
            return [ANY], "province"
        if p and p not in ("Khác", "Nước ngoài"):
            u = units_from_list(p)
            if u:
                return u, "province"
        for c in ("job_address_x", "job_detail_address"):
            u = units_from_text(r.get(c))
            if u:
                return u, "address"
        return [], "unknown"
    loc = df.apply(_loc, axis=1)
    df["units_l"] = loc.map(lambda x: x[0])
    df["loc_source"] = loc.map(lambda x: x[1])

    # ---- occupation / industry / level ----
    df["occupation"] = [occupations_of(t, mi, max_n=1)[0] for t, mi in zip(df["job_title"], df["main_industry"])]
    ig = df["industry_group"].fillna("Khác").astype(str)
    df["industry_group_src"] = ig
    occ_g = df["occupation"].map(OCC_GROUP)
    df["industry_group"] = np.where(df["occupation"].ne("khac"), occ_g, np.where(ig.isin(INDUSTRY_GROUPS), ig, "Khác"))
    lv_func = df["job_function"].map(_FUNC_LEVEL)
    lv_title = df["job_title"].map(lambda t: level_from_text(t, default=-1))
    df["level"] = np.where(lv_title >= 0, lv_title, lv_func.fillna(1)).astype(int)
    df.loc[df["employment_type"].eq("intern") & (df["level"] <= 1), "level"] = 0

    # ---- experience / salary / degree ----
    df["exp_min"] = pd.to_numeric(df["exp_min"], errors="coerce").clip(0, 15)
    df["exp_max"] = pd.to_numeric(df["exp_max"], errors="coerce").clip(0, 20)
    sal = df.apply(job_salary, axis=1)
    df["sal_min"], df["sal_max"], df["sal_kind"] = sal.map(lambda x: x[0]), sal.map(lambda x: x[1]), sal.map(lambda x: x[2])
    df["deg_min"] = df["job_requirement"].map(job_min_degree)

    # ---- benefits ----
    btxt = df["benefits"].fillna("").map(fold)
    df["ben_cats"] = [";".join(str(i) for i, rx in enumerate(_BEN_RE) if rx.search(b)) for b in btxt]
    df["ben_known"] = btxt.str.len() > 2

    # ---- de-duplicate (same posting re-listed) ----
    key = (df["job_title"].map(fold) + "|" + df["company_name"].map(fold) + "|" + df["units_l"].map(lambda l: ";".join(l))
           + "|" + df["job_requirement"].map(lambda s: fold(s)[:200]))
    df["_key"] = key
    ids = df.groupby("_key")["db_job_id"].agg(lambda s: ";".join(map(str, s)))
    df = df.drop_duplicates("_key").copy()
    df["db_job_ids"] = df["_key"].map(ids)
    print(f"[jobs] after de-dup: {len(df)}")

    df["company_posts"] = df.groupby(df["company_name"].map(fold))["db_job_id"].transform("count")
    out = pd.DataFrame({
        "j_idx": np.arange(len(df)),
        "job_id": df["job_id"].astype(str).values,
        "db_job_id": df["db_job_id"].values,
        "db_job_ids": df["db_job_ids"].values,
        "title": df["job_title"].map(clean_text).values,
        "company": df["company_name"].fillna("").map(clean_text).values,
        "province_raw": df["province"].fillna("").values,
        "address": df["job_address_x"].fillna("").map(lambda s: clean_text(s, 200)).values,
        "units": df["units_l"].map(lambda l: ";".join(l)).values,
        "loc_source": df["loc_source"].values,
        "industry_group": df["industry_group"].values,
        "industry_group_src": df["industry_group_src"].values,
        "main_industry": df["main_industry"].fillna("").values,
        "occupation": df["occupation"].values,
        "occupation_name": df["occupation"].map(OCC_NAME).values,
        "job_function": df["job_function"].fillna("").values,
        "level": df["level"].values,
        "employment_type": df["employment_type"].fillna("full_time").values,
        "exp_min": df["exp_min"].values, "exp_max": df["exp_max"].values,
        "sal_min": df["sal_min"].round(2).values, "sal_max": df["sal_max"].round(2).values, "sal_kind": df["sal_kind"].values,
        "deg_min": df["deg_min"].values,
        "ben_cats": df["ben_cats"].values, "ben_known": df["ben_known"].values,
        "company_posts": df["company_posts"].values,
        "deadline": df["deadline"].fillna("").values,
        "benefits": df["benefits"].fillna("").map(lambda s: clean_text(s, 300)).values,
        "requirement": df["job_requirement"].fillna("").map(lambda s: clean_text(s, 500)).values,
        "description": df["job_description"].fillna("").map(lambda s: clean_text(s, 500)).values,
    })
    lvl_name = out["level"].map(LEVELS)
    out["text_full"] = (out["title"] + ". " + out["occupation_name"] + ". " + out["industry_group"] + ". " + out["main_industry"]
                        + ". Cấp bậc: " + lvl_name + ". " + out["requirement"].str[:450] + ". " + out["description"].str[:350])
    return out


# =====================================================================================
# 6. Clean users
# =====================================================================================
def build_users(raw, driver_cap, rng):
    print(f"[users] raw rows: {len(raw)} | unique UserID: {raw['UserID'].nunique()}")
    df = raw.copy()
    df["db_user_id"] = np.arange(1, len(df) + 1)        # id assigned by import_users.py on a fresh DB
    prof_key = df[["User Name", "Desired Job", "Skills", "Target", "Workplace Desired", "Age"]].astype(str).agg("|".join, axis=1).map(fold)
    df["_key"] = df["UserID"].astype(str)
    # rows that are the same profile under different IDs -> same key
    first_id = pd.Series(df["_key"].values, index=prof_key.values).groupby(level=0).first()
    df["_key"] = prof_key.map(first_id).values
    ids = df.groupby("_key")["db_user_id"].agg(lambda s: ";".join(map(str, s)))
    df = df.drop_duplicates("_key").copy()
    df["db_user_ids"] = df["_key"].map(ids)
    print(f"[users] after de-dup: {len(df)}")

    occs = [occupations_of(dj, ind, max_n=2) for dj, ind in zip(df["Desired Job"], df["Industry"])]
    df["occupation"] = [o[0] for o in occs]
    df["occupation2"] = [o[1] if len(o) > 1 else "" for o in occs]

    # cap templated driver profiles (stratified by province so the geography stays intact)
    drv = df.index[df["occupation"].eq("lai_xe")]
    if driver_cap and len(drv) > driver_cap:
        d = df.loc[drv]
        frac = driver_cap / len(d)
        keep = d.groupby("Workplace Desired", group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=int(rng.randint(1e9))))
        if len(keep) > driver_cap:
            keep = keep.sample(driver_cap, random_state=SEED)
        df = df.drop(index=sorted(set(drv) - set(keep.index)))
        print(f"[users] driver profiles capped: {len(drv)} -> {len(keep)}")

    ig = df["industry_group"].fillna("Khác").astype(str)
    df["industry_group_src"] = ig
    df["industry_group"] = np.where(df["occupation"].ne("khac"), df["occupation"].map(OCC_GROUP), np.where(ig.isin(INDUSTRY_GROUPS), ig, "Khác"))
    df["units_l"] = df["Workplace Desired"].map(units_from_list)
    df.loc[df["units_l"].map(len).eq(0), "units_l"] = df.loc[df["units_l"].map(len).eq(0), "province"].map(units_from_list)
    sal = df["Desired Salary"].map(user_salary)
    df["sal_min"], df["sal_max"], df["sal_kind"] = sal.map(lambda x: x[0]), sal.map(lambda x: x[1]), sal.map(lambda x: x[2])
    df["deg_max"] = df["Degree"].map(user_max_degree)
    df["level"] = [level_from_text(dj, default=1) for dj in df["Desired Job"]]
    pt_txt = (df["Desired Job"].fillna("") + " " + df["Industry"].fillna("")).map(fold)
    df["wants_parttime"] = pt_txt.map(lambda s: bool(_PT_USER_RE.search(s)))
    df.loc[df["wants_parttime"] & df["level"].eq(1) & (pd.to_numeric(df["exp_max"], errors="coerce") <= 1), "level"] = 0
    df["married"] = df["Marriage"].fillna("").map(fold).str.contains("da co gia dinh|da ket hon")

    out = pd.DataFrame({
        "u_idx": np.arange(len(df)),
        "user_id": df["UserID"].astype(int).values,
        "db_user_id": df["db_user_id"].values,
        "db_user_ids": df["db_user_ids"].values,
        "name": df["User Name"].fillna("").values,
        "desired_job": df["Desired Job"].fillna("").map(clean_text).values,
        "industry": df["Industry"].fillna("").values,
        "industry_group": df["industry_group"].values,
        "industry_group_src": df["industry_group_src"].values,
        "occupation": df["occupation"].values,
        "occupation2": df["occupation2"].values,
        "occupation_name": df["occupation"].map(OCC_NAME).values,
        "level": df["level"].values,
        "wants_parttime": df["wants_parttime"].values,
        "workplace_raw": df["Workplace Desired"].fillna("").values,
        "units": df["units_l"].map(lambda l: ";".join(l)).values,
        "experience_raw": df["Work Experience"].fillna("").values,
        "exp_min": pd.to_numeric(df["exp_min"], errors="coerce").values,
        "exp_max": pd.to_numeric(df["exp_max"], errors="coerce").values,
        "salary_raw": df["Desired Salary"].fillna("").values,
        "sal_min": df["sal_min"].values, "sal_max": df["sal_max"].values, "sal_kind": df["sal_kind"].values,
        "degree_raw": df["Degree"].fillna("").map(lambda s: clean_text(s, 200)).values,
        "deg_max": df["deg_max"].values,
        "age": pd.to_numeric(df["Age"], errors="coerce").values,
        "gender": df["Gender"].fillna("").values,
        "married": df["married"].values,
        "skills": df["Skills"].fillna("").map(lambda s: clean_text(s, 600)).values,
        "target": df["Target"].fillna("").map(lambda s: clean_text(s, 400)).values,
    })
    out["text_full"] = (out["desired_job"] + ". " + out["occupation_name"] + ". " + out["industry"] + ". " + out["industry_group"]
                        + ". Kỹ năng: " + out["skills"].str[:450] + ". Mục tiêu: " + out["target"].str[:300]
                        + ". Học vấn: " + out["degree_raw"].str[:120])
    return out


# =====================================================================================
# 7. Behaviour model -> simulated applications
# =====================================================================================
SIM = dict(
    w_occ=3.0, w_ind=0.8, w_text=8.0, w_level=0.6,
    loc_base=6.0, exp_per_year=1.0, exp_tol=1.0, deg_per_level=1.4, sal_ratio=0.8, sal_w=3.0,
    w_overqual=0.15, w_attr_sal=0.35, w_attr_ben=0.06, w_attr_company=0.12, w_pop=0.7,
    cf_dim=8, w_cf=0.9, cf_user_noise=0.6,
    temperature=0.4, n_pos_median=20, n_pos_sigma=0.45, n_pos_min=8, n_pos_max=50,
    n_hard=100,
)


def _arrays(jobs, users):
    """Numeric arrays shared by the behaviour model and the hard-negative miner."""
    NU = len(users)
    A = {}
    A["j_occ"] = jobs["occupation"].map(OCC_IDX).values
    A["u_occ"] = users["occupation"].map(OCC_IDX).values
    A["u_occ2"] = users["occupation2"].map(lambda c: OCC_IDX.get(c, -1)).values
    gi = {g: i for i, g in enumerate(INDUSTRY_GROUPS)}
    A["j_ig"] = jobs["industry_group"].map(gi).values
    A["u_ig"] = users["industry_group"].map(gi).values
    A["j_lvl"], A["u_lvl"] = jobs["level"].values.astype(np.float32), users["level"].values.astype(np.float32)
    A["jx"] = jobs["exp_min"].values.astype(np.float32)
    A["ux"] = users["exp_max"].values.astype(np.float32)
    A["jd"] = jobs["deg_min"].values.astype(np.float32)
    A["ud"] = users["deg_max"].values.astype(np.float32)
    A["jh"] = jobs["sal_max"].fillna(jobs["sal_min"]).values.astype(np.float32)
    A["us"] = users["sal_min"].values.astype(np.float32)
    A["jneg"] = jobs["sal_kind"].isin(["negotiable", "missing"]).values

    def mh(series):
        m = np.zeros((len(series), len(UNIT_NAMES)), np.float32)
        anyv = np.zeros(len(series), bool)
        prim = np.full(len(series), -1)
        for i, s in enumerate(series):
            l = [x for x in s.split(";") if x]
            if ANY in l:
                anyv[i] = True
                continue
            for k, x in enumerate(l):
                m[i, UNIT_IDX[x]] = 1
                if k == 0:
                    prim[i] = UNIT_IDX[x]
        return m, anyv, prim
    A["jp"], A["jany"], A["jprim"] = mh(jobs["units"])
    A["up"], A["uany"], A["uprim"] = mh(users["units"])
    A["jhas"], A["uhas"] = A["jp"].sum(1) > 0, A["up"].sum(1) > 0
    A["j_pt"] = jobs["employment_type"].isin(PARTTIME_EMP).values
    A["j_intern"] = jobs["employment_type"].eq("intern").values | (jobs["level"].values == 0)
    A["u_pt"] = users["wants_parttime"].values.astype(bool)
    A["NU"], A["NJ"] = NU, len(jobs)
    return A


def violations(A, ub, exp_tol=1.0, sal_ratio=0.8):
    """Same definitions as the notebook: True = violated; missing data never counts as a violation."""
    overlap = (A["up"][ub] @ A["jp"].T) > 0
    p = (A["uhas"][ub, None] & ~A["uany"][ub, None] & A["jhas"][None] & ~A["jany"][None] & ~overlap)
    with np.errstate(invalid="ignore"):
        e = A["jx"][None] > (A["ux"][ub, None] + exp_tol)
        d = A["jd"][None] > A["ud"][ub, None]
        s = A["jh"][None] < sal_ratio * A["us"][ub, None]
    return dict(province=p, experience=e, degree=d, salary=s)


def build_text_sim(jobs, users):
    from sklearn.feature_extraction.text import TfidfVectorizer
    jt = ((jobs["title"] + " ") * 3 + jobs["main_industry"] + " " + jobs["requirement"].str[:300]).map(fold)
    ut = ((users["desired_job"] + " ") * 3 + users["industry"] + " " + users["skills"].str[:300]).map(fold)
    tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.3, ngram_range=(1, 2), max_features=200000, dtype=np.float32)
    XJ = tv.fit_transform(jt)
    XU = tv.transform(ut)
    return XU, XJ


def simulate_applications(jobs, users, rng, A, XU, XJ):
    """
    Utility of user u for job j (all terms documented in meta.json -> 'behaviour_model'):
      U(u,j) = w_occ * S_occ[occ_u, occ_j]               role fit (related roles get partial credit)
             + w_ind * [industry_u == industry_j]
             + w_text * tfidf(u, j)                         fine-grained title/skill match
             - w_level * |level_u - level_j|
             + attractiveness_j                             salary vs role median, #benefits, employer size,
                                                            latent popularity eps_j (only visible through edges)
             + w_cf * <a_u, b_j>                            latent taste shared by users of the same role
             - lambda_loc(u)  * loc_cost(u,j)               lambda depends on #wished units, age, marital status
             - exp_per_year   * max(0, jx - ux - tol)       - small over-qualification cost for entry jobs
             - deg_per_level  * max(0, jd - ud)
             - lambda_sal(u)  * max(0, log(0.8*us / jh))    only users who stated a salary care
             + employment-type fit (part-time seekers vs full-time jobs and vice versa)
    Each user applies to n_u ~ LogNormal jobs drawn without replacement with Gumbel-top-k
    (Plackett-Luce) at temperature T: realistic, stochastic, but learnable.
    """
    NU, NJ = A["NU"], A["NJ"]
    S_occ = occ_sim_matrix()
    DIST = haversine_matrix()
    DIST_EXT = np.zeros((len(UNIT_NAMES) + 1, len(UNIT_NAMES) + 1), np.float32)
    DIST_EXT[:-1, :-1] = DIST

    # ----- job attractiveness (observable part + latent popularity) -----
    sal_mid = np.log1p(jobs[["sal_min", "sal_max"]].mean(axis=1))
    med = sal_mid.groupby(jobs["occupation"]).transform("median")
    sd = sal_mid.groupby(jobs["occupation"]).transform("std").fillna(0.3).clip(lower=0.15)
    sal_z = ((sal_mid - med) / sd).clip(-2.5, 2.5).fillna(0).values.astype(np.float32)
    n_ben = jobs["ben_cats"].map(lambda s: len([x for x in s.split(";") if x])).values.astype(np.float32)
    eps_pop = rng.standard_normal(NJ).astype(np.float32)
    attract = (SIM["w_attr_sal"] * sal_z + SIM["w_attr_ben"] * n_ben
               + SIM["w_attr_company"] * np.log1p(jobs["company_posts"].values).astype(np.float32) + SIM["w_pop"] * eps_pop)

    # ----- latent collaborative taste -----
    k = SIM["cf_dim"]
    occ_taste = rng.standard_normal((len(OCC_CODES), k)).astype(np.float32)
    a_u = occ_taste[A["u_occ"]] + SIM["cf_user_noise"] * rng.standard_normal((NU, k)).astype(np.float32)
    b_j = (rng.standard_normal((NJ, k)) / math.sqrt(k)).astype(np.float32)

    # ----- per-user constraint sensitivities (from observable attributes) -----
    n_units = A["up"].sum(1)
    age = users["age"].fillna(30).values
    married = users["married"].values.astype(bool)
    lam_loc = SIM["loc_base"] * np.where(n_units >= 2, 0.6, 1.0) * np.where(married | (age >= 35), 1.25, np.where(age < 25, 0.8, 1.0))
    lam_loc = np.where(A["uany"], 0.0, lam_loc).astype(np.float32)
    lam_sal = np.where(np.isnan(A["us"]), 0.0, SIM["sal_w"]).astype(np.float32)

    n_pos = np.clip(np.round(np.exp(np.log(SIM["n_pos_median"]) + SIM["n_pos_sigma"] * rng.standard_normal(NU))
                             * np.where(age < 28, 1.15, 1.0)), SIM["n_pos_min"], SIM["n_pos_max"]).astype(int)

    def utility(ub):
        occ = np.maximum(S_occ[A["u_occ"][ub]][:, A["j_occ"]],
                         np.where(A["u_occ2"][ub, None] >= 0, 0.8 * S_occ[np.maximum(A["u_occ2"][ub], 0)][:, A["j_occ"]], 0))
        ind = (A["u_ig"][ub, None] == A["j_ig"][None]) & (A["j_ig"][None] != INDUSTRY_GROUPS.index("Khác"))
        txt = (XU[ub] @ XJ.T).toarray()
        lvl = np.abs(A["u_lvl"][ub, None] - A["j_lvl"][None])
        # location
        overlap = (A["up"][ub] @ A["jp"].T) > 0
        d_km = DIST_EXT[A["uprim"][ub]][:, A["jprim"]]
        loc_cost = np.where(A["jany"][None] | overlap, 0.0,
                            np.where(~A["jhas"][None] | ~A["uhas"][ub, None], 0.35, 0.6 + 0.4 * (1 - np.exp(-d_km / 200.0))))
        # experience / degree / salary
        with np.errstate(invalid="ignore"):
            short = np.nan_to_num(np.maximum(0, A["jx"][None] - A["ux"][ub, None] - SIM["exp_tol"]))
            over = np.nan_to_num(np.maximum(0, A["ux"][ub, None] - A["jx"][None] - 4)) * (A["j_lvl"][None] <= 1)
            dshort = np.nan_to_num(np.maximum(0, A["jd"][None] - A["ud"][ub, None]))
            sgap = np.nan_to_num(np.maximum(0, np.log(SIM["sal_ratio"] * A["us"][ub, None] / A["jh"][None])))
        sal_unknown_pen = 0.2 * (A["jneg"][None] & ~np.isnan(A["us"][ub, None]))
        # employment type
        upt = A["u_pt"][ub, None]
        emp = np.where(upt, np.where(A["j_pt"][None], 1.2, -0.6), np.where(A["j_pt"][None], -1.0, 0.0))
        emp = emp - 1.2 * (A["j_intern"][None] & (np.nan_to_num(A["ux"][ub, None]) >= 3))
        return (SIM["w_occ"] * occ + SIM["w_ind"] * ind + SIM["w_text"] * txt - SIM["w_level"] * lvl
                + attract[None] + SIM["w_cf"] * (a_u[ub] @ b_j.T) / math.sqrt(k)
                - lam_loc[ub, None] * loc_cost
                - SIM["exp_per_year"] * short - SIM["w_overqual"] * over
                - SIM["deg_per_level"] * dshort
                - lam_sal[ub, None] * sgap - sal_unknown_pen
                + emp).astype(np.float32)

    pos = {}
    B = 128
    for s0 in range(0, NU, B):
        ub = np.arange(s0, min(NU, s0 + B))
        U = utility(ub)
        G = -np.log(-np.log(rng.uniform(1e-9, 1, U.shape))).astype(np.float32)
        Z = U / SIM["temperature"] + G
        for r, u in enumerate(ub):
            n = n_pos[u]
            idx = np.argpartition(-Z[r], n)[:n]
            pos[u] = idx[np.argsort(-Z[r, idx])]
    hidden = dict(eps_pop=eps_pop, a_u=a_u, b_j=b_j, lam_loc=lam_loc, lam_sal=lam_sal, utility=utility)
    return pos, hidden


def mine_hard_negatives(A, pos, XU, XJ, n_hard):
    """Relevant (same/related role or strong text match) but violating a hard constraint."""
    S_occ = occ_sim_matrix()
    hard = {}
    B = 128
    for s0 in range(0, A["NU"], B):
        ub = np.arange(s0, min(A["NU"], s0 + B))
        rel = 2.0 * S_occ[A["u_occ"][ub]][:, A["j_occ"]] + 4.0 * (XU[ub] @ XJ.T).toarray()
        v = violations(A, ub)
        infeas = v["province"] | v["experience"] | v["degree"]
        sc = np.where(infeas & (rel >= 1.5), rel, -1)
        for r, u in enumerate(ub):
            sc[r, pos[u]] = -1
            top = np.argpartition(-sc[r], n_hard)[:n_hard]
            top = top[sc[r, top] > 0]
            hard[u] = top[np.argsort(-sc[r, top])]
    return hard


# =====================================================================================
# 8. Sanity check: simple baselines on the test split (is the task learnable, not trivial?)
# =====================================================================================
def baseline_check(A, pos, split, XU, XJ, hidden, users, jobs):
    NU, NJ = A["NU"], A["NJ"]
    disc = 1.0 / np.log2(np.arange(2, 22))
    train_deg = np.zeros(NJ, np.float32)
    for u in range(NU):
        np.add.at(train_deg, split["train"][u], 1)
    # item-item CF from train edges
    import scipy.sparse as sp
    rows = np.concatenate([[u] * len(split["train"][u]) for u in range(NU)])
    cols = np.concatenate([split["train"][u] for u in range(NU)])
    R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(NU, NJ))
    uu = R @ R.T                                   # user-user co-application
    S_occ = occ_sim_matrix()
    res = {k: dict(ndcg=[], recall=[]) for k in ("random", "popularity", "content_tfidf", "content+feasible", "userKNN_CF", "content+feasible+CF", "ORACLE_true_utility")}
    rng = np.random.RandomState(0)
    B = 128
    for s0 in range(0, NU, B):
        ub = np.arange(s0, min(NU, s0 + B))
        txt = (XU[ub] @ XJ.T).toarray()
        occ = S_occ[A["u_occ"][ub]][:, A["j_occ"]]
        v = violations(A, ub)
        infeas = v["province"] | v["experience"] | v["degree"]
        cont = 3 * occ + 4 * txt
        uu_b = uu[ub].toarray()
        uu_b[np.arange(len(ub)), ub] = 0
        cf = (sp.csr_matrix(uu_b) @ R).toarray()
        scores = {
            "random": rng.rand(len(ub), NJ),
            "popularity": np.tile(train_deg, (len(ub), 1)),
            "content_tfidf": txt,
            "content+feasible": cont - 5 * infeas,
            "userKNN_CF": cf,
            "content+feasible+CF": cont - 5 * infeas + 1.5 * np.log1p(cf),
            "ORACLE_true_utility": hidden["utility"](ub),
        }
        for name, sc in scores.items():
            sc = sc.astype(np.float32)
            for r, u in enumerate(ub):
                sc[r, np.concatenate([split["train"][u], split["val"][u]])] = -1e9
                gt = split["test"][u]
                top = np.argpartition(-sc[r], 20)[:20]
                top = top[np.argsort(-sc[r, top])]
                hit = np.isin(top, gt).astype(float)
                res[name]["recall"].append(hit.sum() / len(gt))
                res[name]["ndcg"].append((hit * disc).sum() / disc[:min(len(gt), 20)].sum())
    out = {k: {m: round(float(np.mean(v_)), 4) for m, v_ in d.items()} for k, d in res.items()}
    print("[check] test Recall@20 / NDCG@20 (ranking all jobs, train+val excluded):")
    for k, d in out.items():
        print(f"   {k:22s} R@20={d['recall']:.4f}  NDCG@20={d['ndcg']:.4f}")
    return out


def write_readme(out_dir, meta):
    bc = meta.get("baseline_check") or {}
    bl = "\n".join(f"| {k} | {v['recall']:.4f} | {v['ndcg']:.4f} |" for k, v in bc.items()) or "| (run with --check) | | |"
    cov = meta["coverage"]
    txt = f"""# VN Job Rec Graph

Training data for the heterogeneous-GNN job recommender (`Final_Project_Kaggle (2).ipynb`).
Built by `Job_Recommended/ml/build_training_dataset.py` (seed {meta['seed']}) from the data the backend serves:
`import_db/COMBINED_DATA_PROCESSED1.csv` (39,036 job rows) and `import_db/USER_DATA_PROCESSED1.csv` (3,983 seeker rows).

| file | rows | content |
|---|---|---|
| `jobs.csv` | {meta['n_jobs']:,} | cleaned postings (`j_idx` = graph index, `db_job_id` = backend id) |
| `users.csv` | {meta['n_users']:,} | cleaned seekers (`u_idx` = graph index, `db_user_id` = backend id) |
| `interactions.csv` | {meta['n_interactions']:,} | simulated applications `u_idx, j_idx, split (train/val/test), pref_rank` |
| `hard_negatives.csv` | {meta['n_hard_negatives']:,} | relevant-but-infeasible jobs per user (province / experience / degree violated) |
| `meta.json` | | taxonomies, unit coordinates, coverage, behaviour-model parameters, baseline check |

## Cleaning
* Jobs: duplicate re-postings removed; unknown province ('Khác', 7%) recovered from the street address
  (districts, industrial parks, English spellings); 63 provinces mapped to the 34 post-2025 units;
  unknown industry recovered via a 35-role occupation taxonomy shared with users; salary in million VND/month
  (`salary_type` 3 = negotiable had imputed numbers -> NaN; 'Trên X' / 'Đến X' keep only the real bound);
  minimum degree parsed from requirement text ('ưu tiên' mentions ignored); level from job_function/title; 9 benefit groups.
* Users: duplicate UserIDs and cloned profiles removed; desired salary parsed from the original text
  (numeric columns were mis-scaled); degree level parsed from the education text; multi-province wishes kept;
  occupation, level and part-time intent extracted; templated driver profiles capped at {meta['driver_cap']}
  (they were 50% of rows but only ~1% of jobs are driving jobs).

Coverage — jobs: location {cov['job']['units']:.1%}, salary {cov['job']['salary']:.1%}, degree {cov['job']['degree']:.1%}, occupation {cov['job']['occupation_known']:.1%};
users: location {cov['user']['units']:.1%}, salary {cov['user']['salary']:.1%}, degree {cov['user']['degree']:.1%}, occupation {cov['user']['occupation_known']:.1%}.

## Labels (simulated — there is no real interaction log)
Each user applies to ~{SIM['n_pos_median']} jobs drawn by Gumbel-top-k (Plackett–Luce, T={SIM['temperature']}) from a utility that combines
role/industry/text fit, job attractiveness (salary vs role median, benefits, employer size, latent popularity),
a latent taste shared by users of the same role (only learnable from the graph), and constraint penalties whose
strength depends on observable user attributes (number of wished provinces, age, marital status, stated salary).
Positives violating a hard constraint: province {meta['positives_violation_rate']['province']:.1%},
experience {meta['positives_violation_rate']['experience']:.1%}, degree {meta['positives_violation_rate']['degree']:.1%}
(random job: {meta['random_violation_rate']['province']:.1%} / {meta['random_violation_rate']['experience']:.1%} / {meta['random_violation_rate']['degree']:.1%}).

## Difficulty check (test split, rank all jobs, train+val excluded)
| ranker | Recall@20 | NDCG@20 |
|---|---|---|
{bl}

`ORACLE_true_utility` uses the hidden utility itself: it is the ceiling, not a model. Report results as
"on simulated interactions"; use the hand-label set exported by the notebook for an independent check.
"""
    open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8").write(txt)


# =====================================================================================
# 9. Main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "kaggle_dataset"))
    ap.add_argument("--driver-cap", type=int, default=500)
    ap.add_argument("--check", action="store_true", help="run baseline sanity check")
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    os.makedirs(args.out, exist_ok=True)

    jobs = build_jobs(load_source("COMBINED_DATA_PROCESSED1.csv"))
    users = build_users(load_source("USER_DATA_PROCESSED1.csv"), args.driver_cap, rng)

    A = _arrays(jobs, users)
    XU, XJ = build_text_sim(jobs, users)
    pos, hidden = simulate_applications(jobs, users, rng, A, XU, XJ)

    # ----- split 70/15/15 per user -----
    split = {"train": {}, "val": {}, "test": {}}
    rows = []
    for u in range(len(users)):
        p = rng.permutation(pos[u])
        nv = max(1, int(round(.15 * len(p))))
        nt = max(1, int(round(.15 * len(p))))
        split["val"][u], split["test"][u], split["train"][u] = p[:nv], p[nv:nv + nt], p[nv + nt:]
        rank = {j: r for r, j in enumerate(pos[u])}       # order of preference in the simulation (1 = most wanted)
        for sp_name in ("train", "val", "test"):
            for j in split[sp_name][u]:
                rows.append((u, int(j), sp_name, rank[j] + 1))
    inter = pd.DataFrame(rows, columns=["u_idx", "j_idx", "split", "pref_rank"])
    inter["user_id"] = users["user_id"].values[inter["u_idx"]]
    inter["job_id"] = jobs["job_id"].values[inter["j_idx"]]

    hard = mine_hard_negatives(A, pos, XU, XJ, SIM["n_hard"])
    hn = pd.DataFrame([(u, int(j)) for u in range(len(users)) for j in hard[u]], columns=["u_idx", "j_idx"])

    # ----- stats: how often do positives violate a constraint (soft behaviour) -----
    vstats = {c: [] for c in ("province", "experience", "degree", "salary")}
    rnd = {c: [] for c in vstats}
    for s0 in range(0, len(users), 256):
        ub = np.arange(s0, min(len(users), s0 + 256))
        v = violations(A, ub)
        for c in vstats:
            for r, u in enumerate(ub):
                vstats[c].append(v[c][r, pos[u]].mean())
            rnd[c].append(v[c].mean(1))
    viol_pos = {c: round(float(np.mean(x)), 4) for c, x in vstats.items()}
    viol_rand = {c: round(float(np.mean(np.concatenate(x))), 4) for c, x in rnd.items()}
    same_occ = float(np.mean([np.mean(A["j_occ"][pos[u]] == A["u_occ"][u]) for u in range(len(users))]))
    print(f"[sim] positives: {len(inter)} | per user median {int(np.median([len(pos[u]) for u in pos]))} | same occupation {same_occ:.1%}")
    print(f"[sim] violation rate among positives: {viol_pos}  vs random job: {viol_rand}")

    check = baseline_check(A, pos, split, XU, XJ, hidden, users, jobs) if args.check else None

    # ----- write -----
    jobs.to_csv(os.path.join(args.out, "jobs.csv"), index=False, encoding="utf-8")
    users.to_csv(os.path.join(args.out, "users.csv"), index=False, encoding="utf-8")
    inter.to_csv(os.path.join(args.out, "interactions.csv"), index=False)
    hn.to_csv(os.path.join(args.out, "hard_negatives.csv"), index=False)

    def _cov(df, col):
        return round(float(df[col].notna().mean()), 4)
    meta = dict(
        name="Vietnamese job recommendation graph dataset (backend data, cleaned + simulated applications)",
        seed=SEED, driver_cap=args.driver_cap,
        n_jobs=len(jobs), n_users=len(users), n_interactions=len(inter),
        split_counts=inter["split"].value_counts().to_dict(), n_hard_negatives=len(hn),
        unit_names=UNIT_NAMES, unit_latlon=[list(UNITS[n][:2]) for n in UNIT_NAMES], any_unit=ANY,
        occupations={c: dict(name=OCC_NAME[c], industry_group=OCC_GROUP[c]) for c in OCC_CODES},
        industry_groups=INDUSTRY_GROUPS, levels=LEVELS, degree_levels=DEGREE_NAMES, benefit_groups=list(BEN_CATS),
        units_note="units = ';'-separated names of the 34 post-2025 units, or 'Toàn quốc'; empty = unknown",
        salary_unit="million VND / month; NaN when negotiable ('thỏa thuận') or missing",
        coverage=dict(
            job=dict(units=round(float(jobs["units"].ne("").mean()), 4), exp=_cov(jobs, "exp_min"),
                     salary=round(float((jobs["sal_min"].notna() | jobs["sal_max"].notna()).mean()), 4), degree=_cov(jobs, "deg_min"),
                     industry_known=round(float(jobs["industry_group"].ne("Khác").mean()), 4),
                     occupation_known=round(float(jobs["occupation"].ne("khac").mean()), 4)),
            user=dict(units=round(float(users["units"].ne("").mean()), 4), exp=_cov(users, "exp_max"),
                      salary=round(float((users["sal_min"].notna()).mean()), 4), degree=_cov(users, "deg_max"),
                      industry_known=round(float(users["industry_group"].ne("Khác").mean()), 4),
                      occupation_known=round(float(users["occupation"].ne("khac").mean()), 4))),
        behaviour_model=dict(params=SIM, description=simulate_applications.__doc__.strip()),
        positives_violation_rate=viol_pos, random_violation_rate=viol_rand, positives_same_occupation=round(same_occ, 4),
        baseline_check=check,
        backend_mapping="db_job_id / db_user_id = row number (1-based) of COMBINED_DATA_PROCESSED1.csv / USER_DATA_PROCESSED1.csv "
                        "= id given by import_jobs.py / import_users.py on a fresh DB; db_*_ids lists all merged duplicates.",
    )
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    json.dump({"title": "VN Job Rec Graph", "id": "YOUR_KAGGLE_USERNAME/vn-job-rec-graph", "licenses": [{"name": "other"}]},
              open(os.path.join(args.out, "dataset-metadata.json"), "w"), indent=1)
    write_readme(args.out, meta)
    print("[done] ->", args.out, {f: f"{os.path.getsize(os.path.join(args.out, f)) / 1e6:.1f} MB" for f in sorted(os.listdir(args.out))})
    print(json.dumps(meta["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
