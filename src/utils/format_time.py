def format_time(t): 

    h = int(t / 3600)
    m = int((t - (h * 3600)) / 60)
    s = int(t - (3600 * h) - (60 * m))

    return f"{h:02d}:{m:02d}:{s:02d}"
