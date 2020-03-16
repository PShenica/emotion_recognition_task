path = "dataset/3901.xml"
count = 0

with open(path) as f:
    xml = f.read()

    for row in xml.splitlines():
        if row.strip()[:4] == "<ki>":
            print(row.strip())
            count += 1

    print(count)

