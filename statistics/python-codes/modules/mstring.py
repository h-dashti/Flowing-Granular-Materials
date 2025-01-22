


###################################################################

def as_si(x, ndp, ignore_one=True):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    
    if ignore_one and m == '1':
        return r'10^{{{e:d}}}'.format(e=int(e))
        
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def show_error(x, err, precision=3):
   return '{0:0.{1}f}({2})'.format(x, precision, round(err*10**precision))