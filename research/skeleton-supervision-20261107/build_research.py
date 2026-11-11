import json, hashlib, uuid, math, re
from pathlib import Path

OUT=Path(__file__).parent
original=json.loads((OUT/'original.canvas').read_text(encoding='utf-8-sig'))
papers={}
def paper(key,title,year,authors,venue,url,evidence,limit):
    papers[key]=dict(title=title,year=year,authors=authors,venue=venue,url=url,evidence=evidence,limit=limit)

paper('Martinez','A Simple yet Effective Baseline for 3D Human Pose Estimation','2017','Julieta Martinez et al.','ICCV','https://openaccess.thecvf.com/content_iccv_2017/html/Martinez_A_Simple_yet_ICCV_2017_paper.html','从2D关节回归3D位置的强基线。','视觉任务；只能作为直接坐标监督的参考，不证明雷达预训练有效。')
paper('VideoPose3D','3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training','2019','Dario Pavllo et al.','CVPR','https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf','半监督重投影与骨长约束；官方run.py有骨长统计匹配实现。','官方训练主项是位置误差；不能因报告速度误差便写成训练用了速度损失。')
paper('RepNet','RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation','2019','Bastian Wandt and Bodo Rosenhahn','CVPR','https://openaccess.thecvf.com/content_CVPR_2019/papers/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.pdf','§3.3用KCS骨向量Gram矩阵表达骨长和关节夹角，送入判别器。','原文是对抗先验；直接以Gram矩阵作蒸馏损失是可借鉴设计，不是其原方法。')
paper('SmoothNet','SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos','2022','Ailing Zeng et al.','ECCV；预印本2021','https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf','§4.3 Eq3–4联合位置L1与真实加速度误差；补充材料§2.6有损失消融。','是姿态后处理网络；不能把加速度匹配写成把加速度压到零，也不保证雷达表征提升。')
paper('LGSIK','LGSIK-Poser: Skeleton-Aware Full-Body Motion Reconstruction from Sparse Inputs','2025','Linhai Li, Jiayi Lin, Wenhui Zhang','AI 6(8),180','https://www.mdpi.com/2673-2688/6/8/180','Eq5明确采用Vel(1)+Vel(3)+Vel(5)，比较预测与真实的跨帧位移。','公式是欧氏范数，不是必然MSE；未据此确认固定跨度组合普遍最优。全文抓取曾429，公式由出版商索引核对。')
paper('SSR','SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction','2025','Linhai Li, Jiayi Lin, Wenhui Zhang','Electronics 14(18),3664','https://www.mdpi.com/2079-9292/14/18/3664','§3.5.2加入多尺度速度项；消融讨论该项及躯干、层次优化。','稀疏VR重建情境；收益不能直接迁移为雷达预训练结论。')
paper('HuMoR','HuMoR: 3D Human Motion Model for Robust Pose Estimation','2021','Davis Rempe et al.','ICCV','https://geometry.stanford.edu/projects/humor/','学习运动先验并在TestOpt中联合估计运动、体形、地面与接触。','接触/地面先验依赖场景与接触可靠性；不是无条件对所有足部施加零速度。')
paper('TransPose','TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors','2021','Xinyu Yi, Yuxiao Zhou, Feng Xu','ACM TOG / SIGGRAPH','https://xinyu-yi.github.io/TransPose/','多阶段关节位置/旋转估计，结合支撑足和根速度恢复全局位移。','LGSIK引用它，但此次未在原文找到固定1/3/5多跨度损失，不能直接归因给TransPose。')
paper('mmPred','mmPred: Radar-based Human Motion Prediction in the Dark','2026','Junqiao Fan et al.','AAAI；预印本2025','https://ojs.aaai.org/index.php/AAAI/article/download/37378/41340','双域历史运动估计：时间域细节与频域趋势；原文Eq11监督DCT系数，Eq12为扩散噪声目标。','未来姿态预测与通用预训练任务不同；不能把整套系统收益归因为单一频域损失。')
paper('Wave2Body','Wave2Body: Rethinking mmWave Human Pose Estimation as Radar-to-Body Token Translation','2026','Bo Liang, Chen Gong, Wei Gao, Chenren Xu','arXiv预印本','https://arxiv.org/abs/2607.18875','独立学习雷达tokenizer、人体tokenizer，再用配对样本学习映射和根定位。','结构先验可产生合理但错误姿态；离散量化与细节精度有权衡，尚不能视为会议录用。')
paper('TacMamba','Tac-Mamba: A Pose-Guided Cross-Modal State Space Model with Trust-Aware Gating for mmWave Radar Human Activity Recognition','2026','作者见出版商页面','Electronics 15(7),1535','https://www.mdpi.com/2079-9292/15/7/1535','§3.1明确视觉输入为2D骨架；含结构教师、3D辅助回归、modality dropout与可信门控。','不是纯3D骨架教师的直接证据；需区分多模态融合与雷达单模态退化/推理实验。')
paper('FMSSL','Free-form motion directed cross-modal self-supervised learning for mmWave radar human motion perception','在线记录2026；卷期2027','Yaoxi Chen et al.','Pattern Recognition 181,114645','https://doi.org/10.1016/j.patcog.2026.114645','出版商摘要说明同步原始雷达与光学骨架、OT软对比、排序保持、masked trajectory modeling。','可读范围为摘要及节选，未核对完整公式或独立消融；卷期为2027年1月，勿直接写成2026会议论文。')
paper('SoftAlign','Soft Temporal Alignment Distillation for Camera-to-mmWave Radar Human Activity Recognition','2026','Jiuqi Yan et al.','arXiv预印本；10月27日','https://arxiv.org/abs/2610.36637','摘要说明2D关键点教师为雷达学生提供片段级软检索蒸馏目标。','表示空间软匹配不是时间戳校准，也不是单调DTW；是2D教师工作，不等同3D监督。')
paper('MMFi','MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing','2023','Jianfei Yang et al.','NeurIPS Datasets and Benchmarks','https://arxiv.org/abs/2305.10345','支持多模态人体姿态与无线感知研究，提供配对数据与官方工具。','动作片段数据不能直接证明自然连续动作边界理解；各坐标、帧率、同步和标定仍需审计。')

# Further verified paper records and canvas sections are loaded below.
paper('AdvMT','AdvMT: Adversarial Motion Transformer for Long-term Human Motion Prediction','2024','Sarmad Idrees, Jongeun Choi, Seokman Sohn','arXiv预印本','https://arxiv.org/html/2401.05018v1','§III-D Eq2含位置、骨长与时间判别器目标。','未给骨长项完整范数公式；位置与骨长仍可能出现零速度预测，不能独立保证运动真实。')
paper('EventEgo','EventEgo3D++: 3D Human Motion Capture from a Head-Mounted Event Camera','2025','Christen Millerdurai et al.','IJCV','https://arxiv.org/abs/2502.07869','Eq4方向余弦；Eq5名为bone length loss但实际是骨向量MSE；Table6评估组合骨损失。','不能将Eq5归为纯标量骨长差，也不能单独归因方向项消除了时序漂移。')
paper('E3D','E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation','2026','Mayur Deshmukh et al.','arXiv预印本','https://arxiv.org/html/2604.08543v1','§4.3 Eq21标量骨长L1，Eq22方向余弦，另有Eq18帧间位移MSE。','未找到仅骨长与骨长+方向的独立消融；采用两项不等于证明组合更优。')
paper('Beyond','BeyondRetarget: Learning Executable Humanoid Motions Directly from Monocular Video','2026','Tianyu Xiong et al.','arXiv预印本','https://arxiv.org/html/2609.29850v1','Eq5对归一化关键语义肢体向量使用余弦偏差；Table3去除方向项后方向误差变大。','直接视频到机器人运动；需形态对齐，不能当成人体姿态估计任务的通用结论。')
paper('Compositional','Compositional Human Pose Regression','2017','Xiao Sun, Jiaxiang Shang, Shuang Liang, Yichen Wei','ICCV','https://arxiv.org/abs/1704.00159','骨向量参数化及沿运动链的组合损失。','启发结构化监督；原始研究不是毫米波预训练。')
paper('Anatomy','Anatomy-aware 3D Human Pose Estimation with Bone-based Pose Decomposition','2020起多版本','Tianlang Chen et al.','arXiv，版本标题有变化','https://arxiv.org/abs/2002.10322','显式分解骨长和骨方向，并由运动链重建关节。已读v2旧标题为Anatomy-aware 3D Human Pose Estimation in Videos。','引用须固定版本；不能混合旧标题、当前标题与未经核实的会议年份。')
paper('Angles','Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction','2015','Ijaz Akhter, Michael J. Black','CVPR','https://pure.mpg.de/view/item_2259888','姿态条件下的关节角可行域先验。','仅核对机构元数据与核心思想，未完整复核正文；不是统一角度阈值。')
paper('MotionBERT','MotionBERT: A Unified Perspective on Learning Human Motion Representations','2023','Wentao Zhu et al.','ICCV；预印本2022','https://arxiv.org/html/2210.06551v4','§3.3 Eq6–8：从加噪/掩码2D恢复3D，联合位置、速度与置信度加权重投影。','位置和速度是未平方L2范数；混合3D标注训练，不能一概称完全无监督。')
paper('MAMP','Masked Motion Predictors are Strong 3D Action Representation Learners','2023','Yunyao Mao et al.','ICCV','https://arxiv.org/html/2308.07092v1','§3.4 Eq4–6按运动强度构造遮蔽概率；运动越大越可能被遮蔽，目标是运动差分预测。','多遮蔽意味着少可见；仅骨架任务验证，如何映射雷达token仍需实验。')
paper('SkeletonMAE1','SkeletonMAE: Spatial-Temporal Masked Autoencoders for Self-supervised Skeleton Action Recognition','2022','Wu et al.','arXiv首稿','https://arxiv.org/abs/2209.02399','Transformer时空遮蔽关节坐标重建；为MAMP的对照。','与Yan等人的同名图模型不同；此处不指定未核实的会议年份。')
paper('SkeletonMAE2','SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training','2023','Yan et al.','ICCV','https://openaccess.thecvf.com/content/ICCV2023/html/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.html','图编解码器与拓扑感知关节/边重建。','与Wu等人同名但不同论文；骨架内预训练不等于跨模态蒸馏。')
paper('MotionMAE','Self-supervised Video Representation Learning with Motion-Aware Masked Autoencoders','2022首稿；机构记录2024','Yang et al.','arXiv；会议记录见作者机构','https://arxiv.org/abs/2210.04154','RGB视频patch重建与相邻帧像素差预测。','跨领域启发；不是3D骨架运动引导遮蔽的直接证据。')
paper('PoseConv3D','Revisiting Skeleton-Based Action Recognition','2022','Haodong Duan et al.','CVPR','https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html','PoseConv3D将2D关节热图沿时间堆叠。','3D heatmap volume指T×H×W，不是XYZ人体三维体素真值。')
paper('PSKD','Progressive Cross-modal Knowledge Distillation for Human Action Recognition','2022','Ni et al.','ACM MM','https://arxiv.org/abs/2208.08090','骨架教师向智能手表加速度学生渐进蒸馏，推理只使用传感器。','提供跨模态先例；学生是惯性传感器，不能直接论证毫米波收益。')

sections=[]
def section(name,summary,color,items,old=None):
    sections.append(dict(name=name,summary=summary,color=color,items=items,old=old))
def item(title,body,refs=(),old=None,detailold=None):
    return dict(title=title,body=body,refs=list(refs),old=old,detailold=detailold)

section('0｜监督对象与标签条件','先决定骨架是预测目标、训练教师还是采样依据。最终目标是雷达表征与下游行为理解，而非默认追求最低姿态误差。','5',[
item('直接坐标回归与全局/局部解耦','预测P∈R^(T×J×3)。位置主项可用平均关节欧氏距离、L1或Huber。根相对姿态Q_j=P_j−P_root与全局根轨迹应分别监督；只有根相对骨架时不宣称学到了全局平移。',['Martinez','VideoPose3D']),
item('骨架标签不等于精确真值','视觉三角化/姿态估计骨架包含遮挡、缺失、时间偏移。按有效关节和置信度加权，并以有效权重和归一化。骨架辅助训练仍需要同步配对数据，只是可能不需要动作类别标签。',['MotionBERT','MMFi']),
item('表示与坐标约定','统一关节拓扑、左右肢体、m/mm、父→子方向、世界/相机/雷达坐标、帧率。2D重投影要求相机模型；XYZ体素监督不能引用T×H×W热图当作等价证据。',['PoseConv3D'])])

section('1｜骨骼几何约束','长度、方向、骨向量、运动链与关节角是相关但不同的约束。所有方法均须先统一坐标系。','4',[
item('1、骨骼长度约束','b_j=P_j−P_parent(j)，l_j=||b_j||₂。标量长度监督可写L_len=mean|l̂−l|；另有同一受试者跨帧一致性、左右对称先验。不要强制所有受试者相同骨长；AdvMT未明确范数，不擅自替其写成MSE。',['AdvMT','VideoPose3D'],'caea65363c3c2aaf'),
item('2、骨骼方向余弦约束','L_dir=mean[1−(b̂·b)/(||b̂||₂||b||₂)]，仅对有效非零骨段计算，数值实现加ε。朝向依赖参考坐标系；它不是相邻骨段之间的关节夹角。',['E3D','Beyond'],'499e194d54ddc5df','ff99a7fbb1d09317'),
item('3、骨向量与运动链组合监督','L_vec=mean||b̂−b||₂²。数学分解：||b̂−b||²=(l̂−l)²+2l̂l(1−cosθ)。因此骨向量MSE已同时约束长度与朝向。EventEgo3D++ Eq5虽称bone length loss，实际为骨向量MSE。',['EventEgo','Compositional']),
item('4、解剖先验与关系矩阵','显式骨长/方向解耦、前向运动学、姿态条件角度可行域。B为骨向量矩阵，K=BᵀB含长度和夹角；K对整体旋转不变。用K做监督是可借鉴设计，RepNet本身用KCS判别器。',['Anatomy','Angles','RepNet'])],'830e2d84091f2cc2')

section('2｜运动时序约束','“匹配真实运动”与“惩罚运动本身”必须区分；速度、加速度应考虑真实时间间隔。','2',[
item('1、一阶速度 / 位移匹配','v_t=(P_(t+1)−P_t)/Δt；L_vel=mean ρ(v̂−v)。MotionBERT Eq6用未平方L2范数；E-3DPSM Eq18用帧间位移MSE。不能把所有文献都概括成MSE，也不能把评估MPJVE当成训练目标。',['MotionBERT','E3D'],'96c3266887762f4b','e43533dd8ce8daf6'),
item('2、多跨度速度 / 位移损失','LGSIK Eq5：Σ_(s∈{1,3,5}) mean||(P̂_(t+s)−P̂_t)−(P_(t+s)−P_t)||₂。原式未除sΔt，本质为多跨度位移匹配。若统一成速度需除sΔt并重新确定权重；跨度应按秒报告。',['LGSIK','SSR'],'840facbfcbae62bd','be0adcb1c13e4f95'),
item('3、加速度匹配（二阶差分）','a_t=(P_(t+1)−2P_t+P_(t−1))/Δt²。SmoothNet匹配â与真实a，不是强制â→0。原“二阶速度损失”改为加速度损失；差分放大标签噪声，需结合位置主项、缺失掩码与可靠标签。',['SmoothNet'],'cadd7ba715fb0589','2b02e00bdf0c3dc4'),
item('4、频域趋势与接触先验','DCT/频域目标可表达长时趋势，接触约束可限制支撑期脚滑。避免把高频动作一概当噪声；脚部零速仅用于可信接触区间。若最终是在线系统，还需单独验证因果性、延迟与未来帧泄漏。',['mmPred','HuMoR','TransPose'])],'120deee2f30d1ad5')

section('3｜自监督骨架学习','骨架内预训练可产生教师或运动目标；从骨架方法迁移到雷达是另一个需要验证的问题。','6',[
item('1、随机掩码与结构重建','区分两篇SkeletonMAE：Wu等人的Transformer时空关节重建；Yan等人的图模型关节/边重建。MotionBERT则从受损2D恢复3D，训练混合了3D标注监督。',['SkeletonMAE1','SkeletonMAE2','MotionBERT']),
item('2、运动先验引导掩码策略','MAMP明确：运动显著处提高被遮蔽概率，促使模型预测被遮蔽部位的运动差分。依据§3.4 Eq4–6：π=Softmax(I/τ)，通过Gumbel Top-K选择遮蔽token。',['MAMP'],'c8614d8b1964a031'),
item('3、“密集掩码”的准确含义','MAMP中：大运动→更多遮蔽→更少可见；不是大运动处密集保留。迁移雷达时需比较随机、多遮蔽、多保留，固定总遮蔽率和训练预算。骨架关节不能未经映射直接对应雷达点。',['MAMP'],'084d045eec56b8eb'),
item('4、重建什么决定学到什么','关节坐标、骨向量、运动差分和教师潜特征可作不同目标。MotionMAE的输入为RGB视频，它的像素差目标只是跨领域启发，不是骨架掩码策略的证据。',['MotionMAE'])],'eb1e9f472eb66af1')

section('4｜跨模态教师与表示迁移','明确训练时使用哪些模态、推理是否仅雷达；特征相似不能替代对信息可观测性的判断。','3',[
item('1、特征 / 关系 / 软目标蒸馏','骨架教师输出表征，学生预测同维投影特征或关系矩阵；教师应冻结或用明确动量机制，防止共同坍塌。PSKD提供骨架→惯性传感器先例；Tac-Mamba视觉输入实际为2D骨架。',['PSKD','TacMamba']),
item('2、配对、对比与时间对应','FMSSL用同步雷达与光学骨架进行软对比、排序及轨迹建模。Soft Alignment用2D骨架教师为学生片段检索表征，不是硬件同步或单调时序配准。同步误差仍应独立审计。',['FMSSL','SoftAlign']),
item('3、人体token与结构化目标空间','Wave2Body先分别学习雷达与人体token，再学习翻译和根位置。提示可将人体先验与雷达表征解耦；但人体合理不等于观测正确，不能只看骨长或外观合理性。',['Wave2Body'])])

section('5｜面向毫米波的研究假设','以下为基于文献与现有笔记的研究设计，不是已证实方法或新颖性结论。优先小规模、同预算消融。','1',[
item('H1｜全局运动 + 局部运动关系','从同步骨架提取根运动、四肢相对躯干速度、左右部位协同。关系目标较绝对坐标少依赖体形，但也可能丢失方向或绝对位置信息。先与Pose、普通速度、Gram关系基线比较。',['RepNet','PSKD']),
item('H2｜雷达可观测运动监督','标定P_r=R·P_c+t，速度v_r=R·v_c（固定外参）。视线u=P_r/||P_r||，径向速度v_rad=uᵀv_r；符号需对齐设备约定。作为近似运动目标，不将关节视作真实散射中心，也不假设所有骨架点都可见。',['Wave2Body','MMFi']),
item('H3｜可靠性与运动变化','权重结合骨架置信度、缺失、回波支持等，归一化并设最低有效样本门槛，防止将难样本全置零。多尺度ΔM与协同关系用于连续运动变化探针；变化大不等于语义事件边界。',['mmPred','TacMamba']),
item('H4｜运动引导雷达遮蔽','用同步骨架运动映射时间片/身体区域，再决定雷达token遮蔽；RD图与点云对应方式不同。必须检验训练中依赖骨架的遮蔽器在测试时是否被移除，避免标签泄漏。',['MAMP','FMSSL'])])

section('6｜验证路径与文献审计','支持与不支持都保留。论文采用某项、组合消融有效、单项因果贡献是三种不同证据。','5',[
item('最小实验矩阵','A随机初始化；B坐标回归；C=B+骨长/方向；D=B+速度；E=D+加速度/多跨度；F教师蒸馏；G径向目标；H=G+可靠性；I固定预算掩码对照。不要一次加全部模块。所有组共享主干、split、训练样本、步数与调参预算。'),
item('评估表征，而不只评估姿态','姿态：MPJPE、根轨迹、骨长/角度误差；时序：速度和加速度误差，注明m/s或mm/frame。预训练：冻结线性探针、少标签迁移、跨人/跨环境Macro-F1；连续任务另测边界F1、时序描述与延迟。重复种子并报告方差。',['SmoothNet','MMFi']),
item('防泄漏与可复现','按受试者/会话或原始长序列切分后再滑窗；相邻重叠片段不能跨split。归一化统计仅训练集。空间标定、时钟偏移、缺失/遮挡标签需先核查；失败记录和原始数据均保留。'),
item('原图结论修正记录','保留AdvMT骨长节点；EventEgo3D++改为骨向量MSE+方向；E-3DPSM删去未经独立消融支持的“组合优于单项”；BeyondRetarget注明机器人任务；二阶速度改为加速度；MAMP补全遮蔽概率定义。')])

nodes=[]; edges=[]; used=set()
def uid(): return uuid.uuid4().hex[:16]
def node(text,x,y,w,h,color=None,id=None):
    id=id or uid(); assert id not in used; used.add(id)
    n=dict(id=id,type='text',text=text,x=int(x),y=int(y),width=int(w),height=int(h))
    if color: n['color']=color
    nodes.append(n); return id
def edge(a,b,label=None):
    e=dict(id=uid(),fromNode=a,fromSide='right',toNode=b,toSide='left')
    if label: e['label']=label
    edges.append(e)
root=node('## 3D骨架如何做监督？\n面向毫米波雷达表征预训练\n调研核验：2026-11-07',-1250,0,340,220,'5','8b156791e3a4e84b')
question=node('### 1、如何使用3D骨架做监督？\n三种角色：预测目标 / 教师 / 学习策略依据',-800,0,360,200,None,'7ae80c4965e34f28')
edge(root,question)
hub=node('### 现有方法与研究路径\n蓝/绿：已有方法\n红：本项目待验证假设\n正文与参考资料见配套调研笔记',-330,0,330,210,'5','039bbddbc27560bb');edge(question,hub)
readme=node('### 阅读说明\n每条方法右侧列论文完整标题、年份、来源。\n【核实】表示已找到对应原文/元数据；不代表已复现实验。\n研究假设与论文已有结论分开。\n完整审计与限制：[[3D骨架监督调研_2026-11-07]]',-800,-420,800,300,'5')
y=0; citation_old={'EventEgo':'57fe1b94f13bdabf','E3D':'df1c95790c5137c9','Beyond':'e569a616637bcbf0'}; cited_old=set()
for s in sections:
    start=y; positions=[]
    for it in s['items']:
        height=max(300,150+math.ceil(len(it['body'])/43)*27)
        a=node('### '+it['title'],500,y,380,150,s['color'],it['old'])
        b=node(it['body'],980,y,680,height,None,it['detailold']);edge(a,b)
        refs=it['refs']
        if refs:
            text='\n\n'.join(f"**[{k}] {papers[k]['title']}**\n{papers[k]['year']} · {papers[k]['venue']}\n[论文 / 原始来源]({papers[k]['url']})\n{papers[k]['evidence']}\n边界：{papers[k]['limit']}" for k in refs)
            rh=max(height,70+sum(140+math.ceil((len(papers[k]['evidence'])+len(papers[k]['limit'])+len(papers[k]['title']))/65)*22 for k in refs))
            old=next((citation_old[k] for k in refs if k in citation_old and k not in cited_old),None)
            if old:
                cited_old.add(next(k for k in refs if k in citation_old and citation_old[k]==old))
            c=node(text,1800,y,1030,rh,None,old);edge(b,c)
            height=max(height,rh)
        positions.append((a,y));y+=height+100
    cat=node('## '+s['name']+'\n\n'+s['summary'],60,(start+y-100)/2-140,340,280,s['color'],s['old']);edge(hub,cat)
    for a,_ in positions:edge(cat,a)
    y+=150
# Preserve any original IDs not used above as individually auditable notes.
for n in original['nodes']:
    if n['id'] not in used:
        assert n['id']=='e569a616637bcbf0', n['id']
        p=papers['Beyond'];anchor=next(a for a in nodes if a['id']=='ff99a7fbb1d09317')
        extra=node('**'+p['title']+'**\n\n'+p['evidence']+'\n'+p['limit']+'\n[论文]('+p['url']+')',3000,anchor['y'],1030,280,None,n['id']);edge('df1c95790c5137c9',extra)
for n in nodes:
    if n['id'] in (root,question,hub): n['y']=int(y/2)
    if n['id']==readme: n['y']=int(y/2)-420
canvas=dict(nodes=nodes,edges=edges)
(OUT/'思维导图.canvas').write_text(json.dumps(canvas,ensure_ascii=False,indent=2),encoding='utf-8')

report=['# 3D骨架监督调研：面向毫米波雷达表征预训练','核验日期：2026-11-07。范围：从导图的几何、时序、自监督三个分支，扩展到教师蒸馏、雷达适配和实验验证。保留原有主题与节点ID，重新布局。文档是研究综述与实验建议，不是已运行结果。','## 主要判断','骨架可以监督坐标、几何关系、运动变化和潜特征，也能控制遮蔽策略。应先确认要提高姿态恢复还是下游雷达行为理解；不同目标不能只靠同一个姿态指标判定。几何与时序损失已有充分先例，潜在研究点是如何选择与雷达观测相符的结构知识，并在同预算实验中证明迁移收益。','文献状态：会议/期刊与预印本分列；元数据核对、摘要核对、正文公式核对并不等价。没有运行论文代码或本项目训练，也不声称完成穷尽式检索或证明方案新颖性。']
for s in sections:
    report+=['## '+s['name'],s['summary']]
    for it in s['items']:
        report+=['### '+it['title'],it['body']]
        for k in it['refs']:
            p=papers[k];report.append(f"- [{k}] [{p['title']}]({p['url']})（{p['year']}，{p['venue']}）：{p['evidence']} **限制**：{p['limit']}")
report+=['## 公式的实现约定（本文统一写法，不冒充每篇论文原式）',r'设 P 为真实关节，P̂ 为预测，b=P_child−P_parent，ρ 为可选择的 L1/L2/Huber。带置信度目标应按 Σwρ(error)/(Σw+ε) 归一化；无效关节的骨段与相邻差分同时剔除。',r'速度差分除以实际 Δt，加速度除以 Δt²。若直接使用帧差需明确单位是位置/帧；变帧率或多跨度之间不能直接比较。匹配 ΔP̂−ΔP 不等于单纯惩罚 ||ΔP̂||，后者偏好静止。',r'方向项对非零向量计算；父子顺序颠倒会令方向符号翻转。归一化KCS或余弦关系可弱化骨长与全局旋转影响，但也会丢失部分绝对方位信息。', '## 推荐执行顺序与停止条件','1. 审计坐标/同步/置信度，按受试者与会话切分；建立随机初始化和坐标监督基线。\n2. 依次加入骨长、骨方向、速度，先逐项消融，再研究组合；之后比较加速度与多跨度。\n3. 在同预算下加入教师蒸馏与固定总率的掩码策略对照。\n4. 外参和雷达速度符号核验通过，才比较普通速度与径向目标；若无法核验，仅完成无需外参的基线并报告证据不足。\n5. 以冻结编码器探针和跨受试者泛化检验预训练价值；连续行为另用连续数据评估，避免把片段分类收益替代边界理解证据。','## 完整论文索引（标题、作者、年份、来源）']
for k,p in papers.items():
    report.append(f"### [{k}] {p['title']}\n\n- 作者：{p['authors']}\n- 年份/状态：{p['year']}；{p['venue']}\n- 原始来源：[论文 / 元数据]({p['url']})\n- 可用证据：{p['evidence']}\n- 引用边界：{p['limit']}")
report+=['## 检索与可访问性记录','使用原论文/arXiv、CVF/ECVA、作者项目页、出版商与研究机构记录。检索沿原图已有论文向骨向量表示、时序损失、掩码骨架建模和骨架→传感器迁移扩展。Firecrawl CLI不可用，改用联网检索与原文。LGSIK出版商全文请求出现429，使用出版商可索引公式和参考文献；角度先验只核对机构记录。FMSSL只访问摘要与节选，不将其当作公式级证据。','校正项：EventEgo3D++损失命名与公式不一致；E-3DPSM缺少所声称独立消融；MotionBERT的速度项不是MSE；MAMP遮蔽方向有明确原文支持；两篇SkeletonMAE须分开；MotionMAE是RGB方法；PoseConv3D体积是T×H×W；2D教师不能混称3D教师。','本调研给出的径向/关系/变化目标与实验排列是研究建议；没有引用论文证明它们在当前项目有效。']
(OUT/'3D骨架监督调研_2026-11-07.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
(OUT/'papers.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2),encoding='utf-8')
ids=[n['id'] for n in nodes]; assert len(ids)==len(set(ids))
assert all(e['fromNode'] in used and e['toNode'] in used for e in edges)
assert all(n['id'] in used for n in original['nodes'])
overlaps=[]
for i,a in enumerate(nodes):
    for b in nodes[i+1:]:
        if a['x']<b['x']+b['width'] and a['x']+a['width']>b['x'] and a['y']<b['y']+b['height'] and a['y']+a['height']>b['y']:
            overlaps.append((a['id'],b['id']))
assert not overlaps, overlaps
print(json.dumps(dict(papers=len(papers),nodes=len(nodes),edges=len(edges),original_nodes_preserved=len(original['nodes']),overlaps=len(overlaps),source_sha256=hashlib.sha256((OUT/'original.canvas').read_bytes()).hexdigest()),ensure_ascii=False))
