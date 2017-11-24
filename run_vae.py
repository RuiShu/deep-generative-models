import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src',    type=str,   default='svhn',    help="Src data")
parser.add_argument('--trg',    type=str,   default='extra',   help="Trg data")
parser.add_argument('--design', type=str,   default='v1',      help="design")
parser.add_argument('--gw',     type=float, default=.01,       help="Gen weight")
parser.add_argument('--rw',     type=float, default=0.5,       help="Rec weight")
parser.add_argument('--npc',    type=int,   default=None,      help="NPC")
parser.add_argument('--lr',     type=float, default=3e-3,      help="Learning rate")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
codebase_args.args = args = parser.parse_args()
pprint(vars(args))

from codebase.models.vae import vae
from codebase.train import train
from codebase.utils import get_data

# Make model name
setup = [
    ('model={:s}',  'vae'),
    ('src={:s}',  args.src),
    ('trg={:s}',  args.trg),
    ('design={:s}', args.design),
    ('gw={:.0e}', args.gw),
    ('rw={:.0e}', args.rw),
    ('npc={}',  args.npc),
    ('lr={:.0e}',  args.lr),
    ('run={:04d}',   args.run)
]

model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = vae()
M.sess.run(tf.global_variables_initializer())
src = get_data(args.src, npc=args.npc)
trg = get_data(args.trg)
saver = tf.train.Saver()

train(M, src, trg,
      saver=saver,
      has_disc=False,
      add_z=False,
      model_name=model_name)
