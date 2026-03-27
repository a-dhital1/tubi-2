import tiktoken

enc = tiktoken.get_encoding("o200k_base")
lo =  enc.encode("young buck sktwotrapy")
print("encoded:", lo)
print("Decoded:", enc.decode(lo))
