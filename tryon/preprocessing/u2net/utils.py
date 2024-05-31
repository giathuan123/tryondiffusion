def normPRED(d):
    ma = d.max()
    mi = d.min()

    dn = (d - mi) / (ma - mi)

    return dn
