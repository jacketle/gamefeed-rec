"""Paper-style GameFeed schematics; tensor cells and neurons are symbolic.
Python/matplotlib drawing; dimensional labels and connections follow project code.
"""
from pathlib import Path
import base64
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/release/architecture'
DEST=ROOT/'deliverables'
OUT.mkdir(parents=True,exist_ok=True)
DEST.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['SimSun','DejaVu Sans'],
 'font.size':13,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,
 'axes.unicode_minus':False,'savefig.facecolor':'white','figure.facecolor':'white'})
INK='#202632'; EDGE='#324663'
C={'blue':'#DCE4F3','purple':'#EDCFE9','peach':'#F6D0B8','green':'#CEE1B3',
   'yellow':'#FFE89F','gray':'#DEDEDE','white':'#FFFFFF'}
SHADES={'blue':['#DCE7F5','#BCD0EA','#8FABD2','#ABC9E4'],
 'purple':['#EAD6ED','#D1B2DA','#E4C9E9','#B998C6'],
 'green':['#E8F0DB','#C9DFAF','#9FC17D','#DBE9C8'],
 'yellow':['#FFF5CE','#FFE693','#F5CC55','#FFEDB5'],
 'peach':['#FBE5D7','#E9B99D','#F4D1BA','#DBA07E']}
TEXT_NODES=[]; PAGES=[]

def canvas(height=8.8):
 fig,ax=plt.subplots(figsize=(16,height));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
 ax.set(xlim=(0,16),ylim=(0,height));ax.axis('off');return fig,ax

def label(ax,x,y,s,size=13,**kw):
 kw.setdefault('ha','center');kw.setdefault('va','center')
 return ax.text(x,y,s,fontsize=size,color=INK,linespacing=1.25,zorder=8,**kw)

def block(ax,x,y,w,h,s='',color='blue',size=13,rounding=.10):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',
  facecolor=C.get(color,color),edgecolor=EDGE,linewidth=1,zorder=3);ax.add_patch(p)
 if s:TEXT_NODES.append((ax.figure,p,label(ax,x+w/2,y+h/2,s,size)))
 return p

def route(ax,points,dashed=False,color=INK,arrow=True,lw=.9):
 for a,b in zip(points[:-2],points[1:-1]):
  ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,ls='--' if dashed else '-',zorder=2)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if arrow else '-',
  mutation_scale=10,color=color,lw=lw,linestyle='--' if dashed else '-',zorder=2,shrinkA=0,shrinkB=1))

def tensor(ax,x,y,w=1.6,h=.28,rows=1,cols=8,color='blue',mask=False):
 for r in range(rows):
  for c in range(cols):
   fill='#F4F4F4' if mask and r==0 else SHADES[color][(r*3+c*2+c//3)%4]
   ax.add_patch(Rectangle((x+c*w/cols,y+r*h/rows),w/cols,h/rows,fc=fill,
    ec='white' if rows>1 else INK,lw=.7,zorder=4))
 ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=INK,lw=.75,zorder=5))

def op(ax,x,y,s,color='yellow',r=.22,size=17):
 ax.add_patch(Circle((x,y),r,fc=C[color],ec=INK,lw=.95,zorder=4));label(ax,x,y,s,size)

def panel(ax,x,y,s):label(ax,x,y,s,16,ha='left',weight='bold')

def mlp(ax,x,y,w=2.7,h=1.15,dims=('81','128','64')):
 columns=[[(x+k*w/2,y+h*(j+.5)/n) for j in range(n)] for k,n in enumerate((4,6,4))]
 for left,right in zip(columns[:-1],columns[1:]):
  for a in left:
   for b in right:ax.plot([a[0],b[0]],[a[1],b[1]],color='#A1ABBB',lw=.45,zorder=1)
 for k,pts in enumerate(columns):
  for xx,yy in pts:ax.add_patch(Circle((xx,yy),.065,fc=SHADES['blue'][k],ec=EDGE,lw=.6,zorder=4))
  label(ax,pts[0][0],y-.19,dims[k],11)

def ranked(ax,x,y,symbol,color,w=1.15,h=1.2):
 for j in range(4):
  yy=y+h-j*.27;block(ax,x,yy,w,.23,'',color,rounding=.015)
  label(ax,x+.14,yy+.115,str(j+1),9)
  ax.plot([x+.3,x+w-.12],[yy+.115,yy+.115],color=EDGE,lw=.8,zorder=5)
 label(ax,x+w/2,y+.1,r'$\vdots$',16);label(ax,x+w/2,y+h+.38,symbol,13)

def save(fig,stem,title,caption,notes,sources,book):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for f,p,t in TEXT_NODES:
  if f is fig:
   a,b=p.get_window_extent(renderer),t.get_window_extent(renderer)
   if b.x0<a.x0+1 or b.x1>a.x1-1 or b.y0<a.y0+1 or b.y1>a.y1-1:
    raise ValueError('Text exceeds block: '+t.get_text())
 fig.savefig(OUT/f'{stem}.svg');fig.savefig(OUT/f'{stem}.pdf')
 fig.savefig(OUT/f'{stem}.png',dpi=300)
 preview_dpi=300
 fig.savefig(OUT/f'{stem}_preview.png',dpi=preview_dpi)
 book.savefig(fig);plt.close(fig)
 PAGES.append(dict(stem=stem,title=title,caption=caption,notes=notes,sources=sources))

def overview(book):
 fig,ax=canvas(8.3);panel(ax,.45,7.9,'(i) 三路候选生成')
 block(ax,.45,6.4,2.2,.7,'此前观看达标记录','purple')
 route(ax,[(2.65,6.75),(3.05,6.75)])
 for i,h in enumerate((.9,.68,.48,.32)):
  ax.add_patch(Rectangle((3.3+i*.32,6.35),.22,h,fc=SHADES['yellow'][i],ec=EDGE,lw=.65))
 label(ax,3.9,5.99,'累计次数',12);route(ax,[(4.7,6.75),(5.3,6.75)])
 block(ax,5.3,6.38,1.2,.74,'热门','yellow');route(ax,[(6.5,6.75),(7,6.75)])
 ranked(ax,7,6.08,r'$L_{pop}$','yellow')
 block(ax,.45,3.95,2.2,.8,'训练期\n用户—视频交互','purple')
 route(ax,[(2.65,4.35),(3.05,4.35)]);tensor(ax,3.05,3.75,1.6,1.15,rows=5,cols=6)
 label(ax,3.85,3.42,'视频相似度矩阵',12);route(ax,[(4.65,4.35),(5.15,4.35)])
 block(ax,5.15,3.95,1.35,.8,'ItemCF\n历史加权','blue',12)
 route(ax,[(6.5,4.35),(7,4.35)]);ranked(ax,7,3.68,r'$L_{cf}$','blue')
 block(ax,.45,1.5,2.2,.75,'视频文案 + 类别','peach',12)
 route(ax,[(2.65,1.875),(3.05,1.875)]);tensor(ax,3.05,1.28,1.6,1.15,rows=5,cols=6,color='green')
 label(ax,3.85,.94,'BGE 视频向量库',12);route(ax,[(4.65,1.875),(5.15,1.875)])
 block(ax,5.15,1.48,1.35,.8,'历史均值\n相似度检索','green',12)
 route(ax,[(6.5,1.875),(7,1.875)]);ranked(ax,7,1.21,r'$L_{bge}$','green')
 panel(ax,9,7.9,'(ii) 组成候选集合');block(ax,9,5.05,6.5,2.25,'','yellow',rounding=.28).set_zorder(0)
 block(ax,9.45,6.3,2.5,.57,'热门：前 10 条','yellow',13)
 block(ax,9.45,5.4,2.5,.57,'ItemCF：去重补足','blue',13)
 route(ax,[(8.15,6.75),(8.55,6.75),(8.55,6.58),(9.45,6.58)])
 route(ax,[(8.15,4.35),(8.65,4.35),(8.65,5.68),(9.45,5.68)])
 op(ax,12.65,6.15,r'$\cup$',r=.25)
 route(ax,[(11.95,6.58),(12.32,6.58),(12.32,6.15),(12.4,6.15)])
 route(ax,[(11.95,5.68),(12.32,5.68),(12.32,6.15),(12.4,6.15)])
 tensor(ax,13.45,5.6,1.35,1.1,rows=5,cols=4,color='yellow')
 route(ax,[(12.9,6.15),(13.45,6.15)]);label(ax,14.12,5.3,r'$C_u:\ 100$ 条',12)
 panel(ax,9,4.55,'(iii) 集合内重新排列');block(ax,9,1.2,6.5,2.75,'','blue',rounding=.28).set_zorder(0)
 label(ax,11.75,3.5,'倒数排名融合 RRF',15)
 label(ax,11.8,2.75,r'$s(i)=\sum_{r}\frac{w_r}{60+\mathrm{rank}_r(i)}$',20)
 label(ax,11.8,1.93,r'$w_{cf}=0.8,\quad w_{pop}=0.2,\quad w_{bge}=0.2$',14)
 ranked(ax,14,1.6,'前 10 条','green',w=.95,h=1.1);route(ax,[(13.6,2.75),(14,2.75)])
 route(ax,[(14.12,5.05),(14.12,4.0)])
 route(ax,[(8.15,1.875),(8.55,1.875),(8.55,2.75),(9,2.75)],True)
 label(ax,11.95,.62,'BGE 专属名额为 0；各路排名仍用于 RRF。',12)
 save(fig,'01_pipeline','多路召回与候选融合',
  '左侧产生三份排名，右侧先决定哪些视频入选，再决定这些视频的顺序。热门保留10条，ItemCF去重补足100条；默认展示RRF顺序的前10条。',
  ['RRF对出现在某一路列表中的视频累加该路贡献，权重不需要相加为1。',
   'BGE专属名额为0，但已入选视频可获得BGE排名分数。图中展示正常主路径，ItemCF列表不足时另有补足逻辑。',
   'ItemCF相似度从训练交互计算；热门在逐日回放时只累计此前日期的记录。模型排序与作者/文案限制保留为实验模式。'],
  ['src/gamefeed/baselines.py','src/gamefeed/fusion_experiment.py::quota_list'],book)

def towers(book):
 fig,ax=canvas(9.05);panel(ax,.45,8.65,'(i) 用户历史与兴趣表示');panel(ax,9.25,8.65,'(ii) 视频表示')
 for x,w,s,c in [(.45,1.65,'用户 ID','purple'),(2.55,3.8,'历史视频序列','purple'),(6.75,1.5,'历史长度','peach'),
  (9.25,1.8,'视频 ID','purple'),(11.5,1.8,'标签集合','peach'),(13.75,1.8,'视频时长','peach')]:block(ax,x,.45,w,.6,s,c,13)
 block(ax,.45,1.55,1.65,.5,'Embedding','gray',12)
 block(ax,2.55,1.55,3.8,.5,r'逐条视频编码 $g_\theta$','blue',13)
 block(ax,6.75,1.55,1.5,.5,'标准化','gray',12)
 block(ax,9.25,1.55,4.05,.5,'Embedding Layer','gray',13);block(ax,13.75,1.55,1.8,.5,'标准化','gray',12)
 for x in (1.275,4.45,7.5,10.15,12.4,14.65):route(ax,[(x,1.05),(x,1.55)])
 tensor(ax,.7,2.65,1.15,color='purple');label(ax,1.5,3.35,r'$e_u\in\mathbb{R}^{16}$',15,ha='left')
 tensor(ax,3.55,2.55,1.8,1.2,rows=5,cols=8,mask=True);label(ax,3.27,3.16,r'$H_u$',16)
 label(ax,4.45,2.28,r'$L\times64,\ L\leq20$',13)
 tensor(ax,7.35,2.65,.3,color='peach',cols=1);label(ax,7.8,3.35,r'$\ell_u$',16,ha='left')
 tensor(ax,9.55,2.65,1.2,color='purple');label(ax,10.4,3.35,'32 维',12,ha='left')
 tensor(ax,11.85,2.5,1.1,.9,rows=4,cols=4,color='peach',mask=True)
 tensor(ax,14.5,2.65,.3,color='peach',cols=1);label(ax,14.9,3.35,'1 维',12,ha='left')
 for x in (1.275,4.45,7.5,10.15,12.4,14.65):route(ax,[(x,2.05),(x,2.5 if x in (4.45,12.4) else 2.65)])
 op(ax,4.45,4.1,r'$\mu$',r=.24);route(ax,[(4.45,3.75),(4.45,3.86)]);label(ax,5.8,4.1,'忽略补齐位置',11)
 op(ax,12.4,3.95,r'$\mu$',r=.22);route(ax,[(12.4,3.4),(12.4,3.73)]);label(ax,13.22,3.95,'16 维',11)
 block(ax,.45,4.65,7.8,.42,'特征拼接  64 + 16 + 1 = 81','yellow',13)
 block(ax,9.25,4.65,6.3,.42,'特征拼接  32 + 16 + 1 = 49','yellow',13)
 route(ax,[(4.45,4.34),(4.45,4.65)])
 for x in (1.275,7.5,10.15,14.65):route(ax,[(x,2.93),(x,4.65)])
 route(ax,[(12.4,4.17),(12.4,4.65)])
 mlp(ax,2.95,5.55,3,1.15,('81','128','64'));mlp(ax,10.9,5.55,3,1.15,('49','128','64'))
 route(ax,[(4.45,5.07),(4.45,5.25),(2.75,5.25),(2.75,6.12),(2.88,6.12)])
 route(ax,[(12.4,5.07),(12.4,5.25),(10.7,5.25),(10.7,6.12),(10.83,6.12)])
 label(ax,1.45,6.2,r'用户网络 $f_\phi$',13);label(ax,12.4,6.98,r'视频网络 $g_\theta$',13)
 block(ax,3.5,7.05,1.9,.4,'L2 归一化','green',12);block(ax,11.45,7.25,1.9,.4,'L2 归一化','green',12)
 route(ax,[(6.02,6.12),(6.45,6.12),(6.45,6.88),(4.45,6.88),(4.45,7.05)])
 route(ax,[(13.97,6.12),(14.45,6.12),(14.45,7.05),(12.4,7.05),(12.4,7.25)])
 tensor(ax,3.55,7.9,1.8,color='green');tensor(ax,11.5,7.9,1.8,color='green')
 route(ax,[(4.45,7.45),(4.45,7.9)]);route(ax,[(12.4,7.65),(12.4,7.9)])
 label(ax,3.06,8.04,r'$u$',17);label(ax,13.7,8.04,r'$v_i$',17)
 op(ax,8.35,8.04,r'$\cdot$',r=.22,size=21)
 route(ax,[(5.35,8.04),(8.13,8.04)]);route(ax,[(11.5,8.04),(8.57,8.04)])
 label(ax,8.35,8.52,r'$s(u,i)=u^\top v_i$',17)
 route(ax,[(6.35,1.8),(6.55,1.8),(6.55,4.4),(8.75,4.4),(8.75,6.12),(10.83,6.12)],True,color=EDGE)
 label(ax,8.75,6.6,'共用参数',11)
 save(fig,'02_two_tower','双塔召回的计算结构',
  '从下往上看：历史视频经同一个视频编码器得到矩阵，取有效位置的均值，与用户ID向量、历史长度拼接；两塔各输出64维单位向量，再计算点积。',
  ['μ表示平均。灰色行代表补齐位置，不计入平均；空历史使用可训练默认向量。',
   '虚线表示历史分支和视频塔共用gθ参数。两个输出向量之间的圆点表示点积。',
   'ReLU将负数变为0；Dropout训练时屏蔽10%的中间输出；L2归一化将非零向量长度调整为1。',
   '训练使用1个正例与最多4个曝光负例，温度系数0.1；检索按点积选前100条。'],
  ['src/gamefeed/models.py::ItemEncoder','src/gamefeed/models.py::TwoTower','src/gamefeed/training.py::_batch_loss'],book)

def ranking(book):
 fig,ax=canvas(10.7);panel(ax,.45,10.3,'(ii-a) 两个独立模型');panel(ax,9.0,10.3,'(ii-b) 共享底层、分别输出')
 for x,s in [(2.3,'观看达标'),(6.15,'点赞'),(10.5,'观看达标'),(14.25,'点赞')]:
  label(ax,x,9.8,s+'概率',13);block(ax,x-.65,8.85,1.3,.45,'Sigmoid','green',12)
  route(ax,[(x,9.3),(x,9.58)]);op(ax,x,8.38,r'$z$',r=.16,size=15);route(ax,[(x,8.54),(x,8.85)])
 for x in (2.3,6.15):
  mlp(ax,x-1.05,6.95,2.1,.9,('138','128','64'))
  route(ax,[(x+1.12,7.4),(x+1.37,7.4),(x+1.37,8.05),(x,8.05),(x,8.22)])
  tensor(ax,x-.85,6.18,1.7,.26,color='purple')
  route(ax,[(x,6.44),(x,6.68),(x-1.25,6.68),(x-1.25,7.4),(x-1.12,7.4)])
  label(ax,x+1.05,6.3,r'$x_{watch}$' if x==2.3 else r'$x_{like}$',14,ha='left')
 label(ax,4.22,8.7,'参数不共享',12)
 mlp(ax,11.2,6.95,2.3,.9,('138','128','64'));tensor(ax,11.5,6.18,1.7,.26,color='purple')
 label(ax,13.4,6.3,r'$x_{shared}$',14,ha='left')
 route(ax,[(12.35,6.44),(12.35,6.68),(10.95,6.68),(10.95,7.4),(11.13,7.4)])
 route(ax,[(13.57,7.4),(13.82,7.4),(13.82,8.03),(10.5,8.03),(10.5,8.22)])
 route(ax,[(13.82,8.03),(14.25,8.03),(14.25,8.22)])
 panel(ax,.45,5.5,'(i) 特征构造')
 block(ax,.45,4.65,15.1,.4,'拼接  [ 用户 16 | 视频 32 | 历史 32 | 视频 × 历史 32 | 标签 16 | 场景 8 | 数值 2 ]  = 138 维','yellow',13)
 for x in (2.3,6.15,12.35):route(ax,[(x,5.05),(x,5.67),(x,6.18)])
 xs=[.45,2.65,5.1,8.15,10.7,13.2];ws=[1.7,1.9,2.1,1.9,1.9,2.35]
 names=['用户 ID','候选视频 ID','历史视频 ID','标签集合','展示场景','时长 / 历史长度']
 for x,w,s in zip(xs,ws,names):block(ax,x,.4,w,.55,s,'purple' if x<8 else 'peach',12)
 block(ax,.45,1.45,12.15,.42,'Embedding Layer（视频与历史使用同一张视频向量表）','gray',13)
 block(ax,13.2,1.45,2.35,.42,'标准化','gray',12)
 for x,w in zip(xs,ws):route(ax,[(x+w/2,.95),(x+w/2,1.45)])
 tensor(ax,.65,2.55,1.3,color='purple');label(ax,1.55,3.25,r'$e_u$',15,ha='left')
 tensor(ax,2.85,2.55,1.5);label(ax,3.85,3.08,r'$e_i$',15,ha='left')
 tensor(ax,5.5,2.25,1.3,.9,rows=4,cols=6,mask=True)
 tensor(ax,8.45,2.25,1.3,.9,rows=4,cols=6,color='peach',mask=True)
 tensor(ax,10.95,2.55,1.4,color='peach');label(ax,11.9,3.25,r'$e_{tab}$',15,ha='left')
 tensor(ax,13.65,2.55,1.4,cols=2,color='peach')
 for x,w in zip(xs,ws):route(ax,[(x+w/2,1.87),(x+w/2,2.25 if x in (5.1,8.15) else 2.55)])
 op(ax,6.15,3.65,r'$\mu$');route(ax,[(6.15,3.15),(6.15,3.43)])
 op(ax,9.1,3.65,r'$\mu$');route(ax,[(9.1,3.15),(9.1,3.43)])
 op(ax,4.6,3.75,r'$\odot$',r=.23)
 route(ax,[(3.6,2.83),(3.6,3.75),(4.37,3.75)])
 route(ax,[(5.93,3.65),(5.4,3.65),(5.4,3.75),(4.83,3.75)])
 label(ax,4.85,4.22,'逐项相乘',11,ha='left')
 for x,y in ((1.3,2.83),(3.15,2.83),(4.6,3.98),(6.15,3.87),(9.1,3.87),(11.65,2.83),(14.35,2.83)):
  route(ax,[(x,y),(x,4.65)])
 label(ax,8,.12,'上方是两种独立实验；相同的特征构造方式，不表示两个独立模型共用训练参数。',11)
 save(fig,'03_ranker_comparison','行为预测：特征构造与参数共享',
  '底部将字段变成向量，计算历史均值和视频—历史逐项乘积，拼成138维特征；上部比较两个独立模型和共享底层的双输出模型。',
  ['μ表示平均；⊙表示对应位置相乘；z为线性输出，Sigmoid将其转换为0到1之间的预测值。',
   '独立模型分别学习观看达标与点赞，每个模型都有自己的向量参数和全连接参数。',
   '共享模型共同学习向量和138→128→64网络，损失为两种二分类交叉熵之和，点赞权重1。',
   '当前模型排序为实验模式，默认展示保留融合候选的RRF顺序。'],
  ['src/gamefeed/models.py::ExposureRanker','src/gamefeed/models.py::SharedBottomRanker','src/gamefeed/training.py::train_single_rankers'],book)

def semantic(book):
 fig,ax=canvas(8.8);panel(ax,.45,8.38,'(i) 文本编码：每条视频生成一个向量')
 block(ax,.45,5.0,2.3,.65,'文案 + 类别名称','peach',13)
 for j,s in enumerate(('[CLS]','词元','…','[SEP]')):block(ax,.45+j*.59,6.12,.53,.34,s,'yellow',9,rounding=.01)
 route(ax,[(1.6,5.65),(1.6,6.12)]);label(ax,1.6,6.85,r'$T\leq128$ tokens',13)
 tensor(ax,3.45,5.75,1.4,1.55,rows=6,cols=6,color='yellow');label(ax,4.15,5.4,'词元向量',12)
 route(ax,[(2.75,6.29),(3.45,6.29)])
 for j in range(3):block(ax,5.55+j*.12,5.65+j*.16,3.1,1.45,'','blue')
 label(ax,7.34,6.68,'Transformer',16);label(ax,7.34,6.23,'BGE 文本编码器',13)
 route(ax,[(4.85,6.5),(5.55,6.5)]);label(ax,7.25,5.28,'预训练参数保持不变',12)
 tensor(ax,9.65,5.75,1.45,1.55,rows=6,cols=6)
 ax.add_patch(Rectangle((9.65,7.04),1.45,.26,fill=False,ec='#BD784A',lw=1.8,zorder=7))
 label(ax,10.37,7.7,'CLS 输出',13);route(ax,[(8.9,6.5),(9.65,6.5)])
 route(ax,[(11.1,7.17),(11.62,7.17),(11.62,6.5),(12,6.5)])
 block(ax,12,6.22,1.3,.55,'L2 归一化','green',11)
 tensor(ax,14,5.75,1.4,1.55,rows=6,cols=6,color='green');route(ax,[(13.3,6.5),(14,6.5)])
 label(ax,14.7,7.72,r'$V\in\mathbb{R}^{N\times512}$',15);label(ax,14.7,8.18,'视频向量库',12)
 panel(ax,.45,4.38,'(ii) 从历史生成兴趣向量，再进行相似度检索')
 block(ax,.45,2.1,2.25,.65,'历史视频 ID 序列','purple',12);route(ax,[(2.7,2.42),(3.25,2.42)])
 tensor(ax,3.25,1.85,1.65,1.2,rows=5,cols=7,color='green')
 label(ax,4.08,3.38,r'$H_u\in\mathbb{R}^{L\times512}$',14)
 route(ax,[(14.7,5.75),(14.7,4.85),(8.3,4.85),(8.3,3.35),(4.9,3.35),(4.9,2.9)],True)
 label(ax,10.3,4.98,'按视频 ID 查表',11)
 op(ax,5.65,2.42,r'$\mu$',r=.24);route(ax,[(4.9,2.42),(5.41,2.42)])
 block(ax,6.35,2.17,1.4,.5,'L2 归一化','green',11);route(ax,[(5.89,2.42),(6.35,2.42)])
 tensor(ax,8.25,2.28,1.4,color='green');label(ax,8.95,2.94,r'$u\in\mathbb{R}^{512}$',15)
 route(ax,[(7.75,2.42),(8.25,2.42)]);op(ax,10.4,2.42,r'$\times$',r=.24)
 route(ax,[(9.65,2.42),(10.16,2.42)])
 route(ax,[(15.4,6.5),(15.7,6.5),(15.7,3.73),(10.4,3.73),(10.4,2.66)])
 label(ax,12.7,3.87,r'$s=Vu$',17)
 tensor(ax,11.15,1.79,.3,1.3,rows=6,cols=1,color='yellow');route(ax,[(10.64,2.42),(11.15,2.42)])
 block(ax,12.1,2.0,1.45,.85,'可用性过滤\n按分数排序','yellow',11)
 route(ax,[(11.45,2.42),(12.1,2.42)]);ranked(ax,14.3,1.5,'前 100 条','green',w=1.1,h=1.15)
 route(ax,[(13.55,2.42),(14.3,2.42)])
 label(ax,5.9,.83,r'$u=\operatorname{norm}\left(\frac{1}{L}\sum_{j=1}^{L}v_j\right)$',20)
 label(ax,11.5,.78,'没有可用历史 → 热门候选',12)
 save(fig,'04_bge_retrieval','BGE文本编码与历史兴趣聚合',
  '上半图编码视频文案，下半图构造用户兴趣并检索。视频向量提前算好；用户历史只需查表、取平均，再与可用视频计算相似度。',
  ['模型为BAAI/bge-small-zh-v1.5，主实验最多128个token；取CLS输出，归一化得到512维向量。',
   'token是文本切分后的单位；CLS是特殊汇总位置。堆叠矩形表示多层编码器，未按真实层数逐层展开。',
   'N是视频数，L是历史长度（最多20）。V每行对应一个视频向量，Vu一次计算用户与各视频的相似度。',
   '向量库每行来自一条视频的CLS输出，方格颜色用于区分向量和矩阵。'],
  ['src/gamefeed/text_retrieval.py::run_text','docs/BGE_IMPLEMENTATION_REVIEW.md'],book)

def main():
 with PdfPages(DEST/'GameFeed_模型结构图.pdf') as book:overview(book);towers(book);ranking(book);semantic(book)
 sections=[]
 for n,p in enumerate(PAGES,1):
  svg=(OUT/(p['stem']+'.svg')).read_text(encoding='utf-8');svg=svg[svg.index('<svg'):]
  bullets=''.join('<li>'+html.escape(s)+'</li>' for s in p['notes'])
  png='data:image/png;base64,'+base64.b64encode((OUT/(p['stem']+'.png')).read_bytes()).decode()
  svguri='data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()
  sections.append(f'<section id="fig{n}"><div class="heading"><span>FIGURE {n:02d}</span><h2>{p["title"]}</h2><div class="actions"><a download="{p["stem"]}.svg" href="{svguri}">下载 SVG</a><a download="{p["stem"]}.png" href="{png}">下载 PNG</a></div></div><div class="figure" tabindex="0" title="点击放大">{svg}</div><div class="caption"><p>{html.escape(p["caption"])}</p><details><summary>查看符号与实现说明</summary><ul>{bullets}</ul></details></div></section>')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GameFeed · 模型结构图</title><style>
 *{box-sizing:border-box}body{margin:0;background:#f7f7f5;color:#202632;font-family:"Microsoft YaHei",sans-serif}header,main{max-width:1450px;margin:auto;padding:26px 36px}header{padding-top:44px}h1{font-size:28px;margin:0 0 12px}header p{font-size:14px;color:#5c6268}nav{display:flex;gap:20px;flex-wrap:wrap;margin-top:24px}a{color:#294c6b;text-underline-offset:4px}section{margin:12px 0 48px;background:white;border:1px solid #deded9}.heading{display:flex;gap:18px;align-items:center;padding:20px 26px;border-bottom:1px solid #eee}.heading span{font:12px Georgia;letter-spacing:1px;color:#6b7580}h2{font-size:19px;margin:0}.actions{margin-left:auto;display:flex;gap:16px;font-size:12px}.figure{padding:10px;cursor:zoom-in;overflow:auto}.figure svg{display:block;width:100%;height:auto}.caption{padding:0 28px 24px;font-size:14px;line-height:1.8}.caption p{margin:0 0 12px}summary{cursor:pointer;color:#3e5265}li{margin:8px 0}footer{padding:0 36px 40px;max-width:1450px;margin:auto;font-size:13px;color:#626971;line-height:1.8}dialog{border:0;width:96vw;max-width:none;height:94vh;padding:25px}dialog::backdrop{background:#0009}dialog .large{overflow:auto;height:85vh}dialog svg{min-width:1400px;width:100%;height:auto}button{font:inherit;border:1px solid #bbb;background:white;padding:8px 16px;cursor:pointer}.close{position:sticky;top:0;float:right}@media(max-width:700px){header,main{padding:18px 12px}.heading{flex-wrap:wrap;padding:16px}.actions{margin-left:0}.figure svg{min-width:900px}.caption{padding:0 16px 20px}}@media print{header,nav,.actions,footer,details{display:none}main{padding:0;max-width:none}section{border:0;break-after:page;margin:0}.figure{padding:0}.figure svg{min-width:0}.caption{font-size:10px}.heading{padding:8px 0}body{background:white}}
 </style><header><h1>GameFeed · 模型结构图</h1><p>向量、矩阵、网络层与计算关系。图内保留结构，详细解释放在图下。点击任一图放大。</p><nav><a href="#fig1">01 多路融合</a><a href="#fig2">02 双塔召回</a><a href="#fig3">03 行为预测</a><a href="#fig4">04 BGE 文本召回</a></nav></header><main>'''+''.join(sections)+'''</main><footer>画法参考：<a href="https://github.com/HarisIqbal88/PlotNeuralNet">PlotNeuralNet</a>、<a href="https://github.com/alexlenail/NN-SVG">NN-SVG</a>与<a href="https://arxiv.org/html/2506.18382v1">PERSCEN论文结构图</a>。按GameFeed实际代码重新绘制，使用Python / matplotlib导出矢量图。圆形表示运算或神经元，方格表示向量与矩阵；数量为示意，真实维度见标注。</footer><dialog id="zoom"><button class="close" onclick="document.querySelector('#zoom').close()">关闭 ×</button><div class="large"></div></dialog><script>document.querySelectorAll('.figure').forEach(el=>{const open=()=>{document.querySelector('#zoom .large').innerHTML=el.innerHTML;document.querySelector('#zoom').showModal()};el.addEventListener('click',open);el.addEventListener('keydown',e=>{if(e.key==='Enter')open()})});</script></html>'''
 (DEST/'GameFeed_模型结构图.html').write_text(page,encoding='utf-8')
 (OUT/'sources.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Updated four paper-style figures and offline HTML/PDF.')

if __name__=='__main__':main()
