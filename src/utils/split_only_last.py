def split_only_last(s, delimeter = "."): 
    return [item[::-1] for item in s[::-1].split(delimeter, 1)][::-1]