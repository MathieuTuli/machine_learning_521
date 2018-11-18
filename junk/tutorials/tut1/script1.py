def script1():
        # astr = input("1: ")
        # bstr = input("2: ")
        # if astr.isnumeric and bstr.isnumeric:
        #     print(int(astr)+int(bstr))
        # else:
        #     print("error")


        while True:
            astr = input("a: ")
            try:
                aint = int(astr)
                break
            except:
                print("wrong a")
        while True:
            bstr = input("b: ")
            try:
                bint = int(bstr)
                break
            except:
                print("wrong b")

        print(aint+bint)
script1()
