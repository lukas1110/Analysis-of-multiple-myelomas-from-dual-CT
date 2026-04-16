import pydicom

# načítanie súboru
ds = pydicom.dcmread(r"E:\Healthy_DICOM\S56310\DIRFILE")

# vypíše všetky metadáta
print(ds)