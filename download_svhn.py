import subprocess
import os

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    if os.path.exists('data/test_32x32.mat') and os.path.exists('data/train_32x32.mat'):
        print "Using existing data"
    else:
        print "Opening subprocess to download data from URL"
        subprocess.check_output(
            '''
            cd data
            wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
            ''',
            shell=True)

if __name__ == '__main__':
    main()
