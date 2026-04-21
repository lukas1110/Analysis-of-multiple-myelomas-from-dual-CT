import pydicom

# načítanie súboru
ds = pydicom.dcmread(r"E:\Healthy_DICOM\S12760\S203930\I10")

# vypíše všetky metadáta
print(ds)