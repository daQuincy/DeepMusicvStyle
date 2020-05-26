from model import MusicVAE
from loader import load_noteseqs
import numpy as np
import tensorflow as tf
import argparse

tf.reset_default_graph()

ap = argparse.ArgumentParser()
ap.add_argument("-bs", "--batch_size", default=32, type=int)
ap.add_argument("-s", "--save_path", default="vae/", type=str)
ap.add_argument("-e", "--epochs", default=100, type=int)
ap.add_argument("--train_set", default="data/Jsbtr.pkl data/Nmdtr.pkl", type=str)
ap.add_argument("--test_set", default="data/Jsbte.pkl data/Nmdte.pkl", type=str)

ap.add_argument("--x_depth", default="89 33 33", type=str)

ap.add_argument("--enc_rnn", default="hyperlstm", type=str)
ap.add_argument("--enc_rnn_dim", default=512, type=int)
ap.add_argument("--enc_hyper_unit", default=256, type=int)
ap.add_argument("--enc_dropout", default=0.25, type=float)
ap.add_argument("--enc_rnn_layer", default=1, type=int)

ap.add_argument("--dec_rnn", default="hyperlstm", type=str)
ap.add_argument("--dec_rnn_dim", default=512, type=int)
ap.add_argument("--dec_hyper_unit", default=256, type=int)
ap.add_argument("--dec_dropout", default=0.25, type=float)
ap.add_argument("--dec_rnn_layer", default=1, type=int)

ap.add_argument("--attention", default=128, type=int)

ap.add_argument("--cont_dim", default=100, type=int)
ap.add_argument("--cat_dim", default=2, type=int)
ap.add_argument("--style_embed_dim", default=100, type=int)
ap.add_argument("--mu_force", default=2.0, type=float)
ap.add_argument("--gumbel", default=0.67, type=float)

ap.add_argument("--kl_reg", default=1.0, type=float)
ap.add_argument("--kl_anneal", default=1000, type=int)

ap.add_argument("--restore_path", default=None, type=str)

args = ap.parse_args()
x_depth = args.x_depth.split()
x_depth = [int(i) for i in x_depth]

train_set = args.train_set.split()
test_set = args.test_set.split()

train_graph = tf.Graph()
val_graph = tf.Graph()

with train_graph.as_default(): 
    t_it, t_x, t_s, t_l = load_noteseqs(train_set, x_depth, 
                                        batch_size=args.batch_size, augment=True).get_iterator()
    m = MusicVAE(x_depth=x_depth,
                 enc_rnn_dim=args.enc_rnn_dim, enc_hyper_unit=args.enc_hyper_unit, enc_dropout=args.enc_dropout,
                 dec_rnn_dim=args.dec_rnn_dim, dec_hyper_unit=args.dec_hyper_unit, dec_dropout=args.dec_dropout,
                 enc_rnn_layer=args.enc_rnn_layer, dec_rnn_layer=args.dec_rnn_layer,
                 enc_rnn=args.enc_rnn, dec_rnn=args.dec_rnn,
                 attention=args.attention,
                 cont_dim=args.cont_dim, cat_dim=args.cat_dim, mu_force=args.mu_force,
                 gumbel=args.gumbel, style_embed_dim=args.style_embed_dim,
                 kl_reg=args.kl_reg,
                 training=True, beta_anneal_steps=args.kl_anneal)
    m.build(t_x, t_s, t_l, None)
    
with val_graph.as_default(): 
    v_it, v_x, v_s, v_l = load_noteseqs(test_set, x_depth, 
                                        batch_size=20).get_iterator()
    n = MusicVAE(x_depth=x_depth,
                 enc_rnn_dim=args.enc_rnn_dim, enc_hyper_unit=args.enc_hyper_unit, enc_dropout=0.0,
                 dec_rnn_dim=args.dec_rnn_dim, dec_hyper_unit=args.dec_hyper_unit, dec_dropout=0.0,
                 enc_rnn_layer=args.enc_rnn_layer, dec_rnn_layer=args.dec_rnn_layer,
                 enc_rnn=args.enc_rnn, dec_rnn=args.dec_rnn,
                 attention=args.attention,
                 cont_dim=args.cont_dim, cat_dim=args.cat_dim, mu_force=args.mu_force,
                 gumbel=args.gumbel, style_embed_dim=args.style_embed_dim,
                 kl_reg=args.kl_reg,
                 training=False, beta_anneal_steps=args.kl_anneal)
    n.build(v_x, v_s, v_l, None)


tf_config = tf.ConfigProto()
tf_config.allow_soft_placement = True
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config, graph=train_graph)
ss = tf.Session(config=tf_config, graph=val_graph)

if args.restore_path:
    print("[INFO] Restoring from checkpoint {}".format(args.restore_path))
    m.saver.restore(sess, args.restore_path)
else:
    sess.run(m.init)
step = 0
    
tw = tf.summary.FileWriter(args.save_path+"train", sess.graph)
vw = tf.summary.FileWriter(args.save_path+"val", ss.graph)
print("[INFO] Start training...")
for epoch in range(args.epochs):
    sess.run(t_it.initializer)
    train_loss = []
    train_kl = []
    while True:
        try: 
            if (step+1)%20 == 0 or step == 0:
                _, tmp_loss, tmp_kl, step, summ = sess.run([m.op, m.recon_loss, m.kl_loss, m.step, m.summ_op])
                tw.add_summary(summ, step)
            else:
                _, tmp_loss, tmp_kl, step = sess.run([m.op, m.recon_loss, m.kl_loss, m.step])
            train_loss.append(tmp_loss)
            train_kl.append(tmp_kl)
                              
        except tf.errors.OutOfRangeError:
            break
    
    m.saver.save(sess, args.save_path + "vae-epoch{}".format(epoch+1))
    n.saver.restore(ss, args.save_path + "vae-epoch{}".format(epoch+1))
    
    val_loss = []
    val_kl = []
    ss.run(v_it.initializer)
    while True:
        try: 
            
            tmp_loss, tmp_kl, summ = ss.run([n.recon_loss, n.kl_loss, n.summ_op])
            val_loss.append(tmp_loss)
            val_kl.append(tmp_kl)
                              
        except tf.errors.OutOfRangeError:
            vw.add_summary(summ, step)
            break

    train_loss = np.mean(train_loss)
    train_kl = np.mean(train_kl)
    val_loss = np.mean(val_loss)
    val_kl = np.mean(val_kl)
    
    print("{} Train Loss: {:.4f} Train KL: {:.2f}  Val Loss: {:.4f} Val KL: {:.2f}".format(epoch+1, train_loss, train_kl,  val_loss, val_kl))
            