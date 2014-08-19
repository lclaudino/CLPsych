import os.path

def fake_lhood_file(ite, folder):
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    f = open(folder+'/model.lhood','w')
    f.write('Iteration\tlikelihood\titer_time\n') #Header

    for ii in range(ite+1):
        if ii == ite:
            f.write(str(ii)+'\t0\t0')
        else:
            f.write(str(ii)+'\t0\t0\n')

    f.close()

