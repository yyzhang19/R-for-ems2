# For Mac M1pro
# R端安装reticulate
install.packages("reticulate")
library(reticulate)

# 配置Python环境（推荐使用conda）
conda_create("esm_env", python_version = "3.9")

# 在R中执行安装PyTorch arm64专用版
reticulate::conda_install(
  envname = "esm_env",
  packages = c("pytorch", "torchvision", "torchaudio"),
  channel = c("pytorch-nightly", "conda-forge")
)

# 先安装pip
reticulate::conda_install(envname = "esm_env", packages = "pip")

# 安装ESM-2核心库
使用清华镜像加速安装
reticulate::py_install("fair-esm", 
                      envname = "esm_env",
                      pip = TRUE,
                      pip_options = "-i https://pypi.tuna.tsinghua.edu.cn/simple")

use_condaenv("esm_env", required=TRUE)

#下载预训练权重
model_url <- "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
download.file(model_url, destfile = "esm2_t33_650M_UR50D.pt")
#也可以用以下链接手工下载
#https://github.com/facebookresearch/esm?tab=readme-ov-file#pre-trained-models-
tools::md5sum("esm2_t33_650M_UR50D.pt")

library(reticulate)
py_config()  # 验证Python路径
esm <- import("esm")
model <- esm$pretrained$esm2_t33_650M_UR50D()  

#构建R语言接口函数
load_esm2 <- function() {
  esm <- import("esm")
  model <- esm$pretrained$esm2_t33_650M_UR50D()  # 自动处理依赖文件
  return(list(model = model[[1]], alphabet = model[[2]]))
}

#验证安装

#测试模型加载

esm_components <- load_esm2()
print(names(esm_components))  # 应输出 "model" "alphabet"
alphabet<-esm_components[[2]]
model<-esm_components[[1]]
#示例序列处理
# 构建数据转换器
batch_converter <- alphabet$get_batch_converter()
# 输入序列格式
sequences <- list(
  list("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
  list("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")
)
# 转换为模型输入格式
converted_data <- batch_converter(sequences)
batch_tokens <- converted_data[[3]]

#特征提取
# 禁用梯度计算以节省内存
torch <- import("torch")

# 提取第33层特征
with(torch$no_grad(), {
  results <- esm_components$model(batch_tokens, repr_layers = list(33L))
})
embeddings <- results$representations$`33`$numpy()
dim(embeddings)  # 应显示 [1, 序列长度, 1280] # 应显示 [2, 序列长度, 2560]

#结果可视化
# 安装R端可视化包
install.packages("seriation")
library("heatmaply")

# 绘制注意力热图
attention <- results$logits$numpy()
dim(attention[1,,])
heatmaply(attention[1,,], main = "Attention Weights")

#例子
library(bio3d)
# 从PDB文件提取序列
pdb <- read.pdb("1crn.pdb")
seq <- paste(pdb$atom$resid[pdb$calpha], collapse="")

# 生成ESM-2嵌入
esm_embed <- function(sequence) {
  converter <- esm_components$alphabet$get_batch_converter()
  converted <- converter(list(list("pdb_seq", sequence)))
  with(torch$no_grad(), {
    res <- esm_components$model(converted[[3]], repr_layers=list(33L))
  })
  return(res$representations$`33`$numpy()[1,,])
}
embedding_matrix <- esm_embed(seq)
dim(embedding_matrix)
head(embedding_matrix)
