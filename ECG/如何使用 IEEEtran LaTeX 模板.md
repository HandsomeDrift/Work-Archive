# 如何使用 IEEEtran LaTeX 模板

**作者**：IEEE 出版技术部

> **备注**：本手稿创建于 2020 年 10 月，由 IEEE 出版技术部开发。根据 LaTeX 项目公共许可证（LPPL）1.3 版（http://www.latex-project.org/）分发。该许可证副本已包含在 2003/12/01 或之后版本的 LaTeX 发行包文档中。文中观点仅代表作者个人，不提供任何担保，使用者需自行承担风险。

------

## 摘要

本文档描述了最常用的文章结构元素，并介绍了如何使用 IEEEtran LaTeX 类文件来生成适合提交至 IEEE（电气与电子工程师协会）的文档。通过选择合适的类选项，IEEEtran 可用于撰写会议论文、期刊文章和技术简讯（通信类稿件）。

------

## 关键词

类、IEEEtran、LaTeX、论文、排版、模板、格式

------

## 第一节：介绍

欢迎阅读这份更新且简化的 IEEEtran LaTeX 类文件使用文档。IEEE 在分析了数百份作者使用该模板提交的文稿后，整理出了这份易于理解的指南。我们将介绍期刊文章中最常用的结构元素，对于不常见的元素，将引导您参考 “IEEEtran_HOWTO.pdf”。

本文适用于 IEEEtran v1.8b 版本。

IEEEtran 模板包包含以下示例文件：

- `bare_jrnl.tex`（期刊模板）
- `bare_conf.tex`（会议模板）
- `bare_jrnl_compsoc.tex`（计算机学会期刊模板）
- `bare_conf_compsoc.tex`（计算机学会会议模板）
- `bare_jrnl_comsoc.tex`（通信学会期刊模板）

这些是“精简模板”，可帮助用户快速了解文档结构。

本说明假设读者已具备 LaTeX 的基本使用知识。如果你是 LaTeX 新手，建议先阅读 Tobias Oetiker 的《不那么简短的 LaTeX 入门教程》（The Not So Short Introduction to LaTeX），可通过以下链接获取：http://tug.ctan.org/info/lshort/english/lshort.pdf

## 第二节：模板的设计理念与限制

这些模板的设计目的是 **尽可能接近文章/论文的最终外观和页数**。因此，**它们并不是用于生成最终用于印刷或在 IEEE Xplore® 上展示的稿件格式**。

使用这些模板，可以帮助作者大致估算文章在最终排版后会占据的页数。同时，这些 LaTeX 文件的结构设计也方便后续转换为 XML 格式，供 IEEE 的外包排版系统使用。

这些 XML 文件不仅用于生成最终的 PDF 文件（供印刷或上传至 IEEE Xplore®），还会被进一步转换成 HTML 格式，以供在 IEEE Xplore® 网站上展示。

> 你有没有试过查看你的文章在 HTML 格式下的显示效果？

## 第3节：LaTeX 发行版获取方式

IEEE 推荐使用 TeX 用户组（TUG）发布的 LaTeX 发行版，可通过 http://www.tug.org 获取。你可以选择加入 TUG 并索取 DVD 光盘，或通过其网站提供的链接免费下载：http://www.tug.org/texlive/

该 DVD 包含适用于 Windows、macOS 和 Linux 操作系统的发行版本。

------

## 第4节：获取 IEEEtran 模板的方式

**IEEE 模板选择器**（IEEE Template Selector）提供了最及时、最完整的 LaTeX 与 MS Word 模板。请访问：

👉 https://template-selector.ieee.org/

按照页面提示操作，选择适合你投稿期刊或会议的模板。虽然很多期刊都使用标准 IEEEtran LaTeX 模板，但也有一些出版物会使用自定义模板（多数是基于 IEEEtran 的修改版本），这些模板可能包含与本指南略有不同的排版说明。

------

## 第5节：获取 LaTeX 帮助的用户社区

以下几个在线社区对初学者和经验丰富的 LaTeX 用户都非常有帮助。在这些论坛的历史贴文中搜索，可以找到大量常见问题的解答：

- [LaTeX 社区论坛](http://www.latex-community.org/)
- [TeX Stack Exchange 问答社区](https://tex.stackexchange.com/)

------

## 第6节：IEEEtran 的文档类选项

在 LaTeX 文件的开头部分，你需要通过 `\documentclass{}` 设置所使用的文档类型。IEEEtran 提供了以下选项，适用于不同类型的出版物：

- **期刊文章（常规）**

  ```latex
  \documentclass[journal]{IEEEtran}
  ```

- **会议论文**

  ```latex
  \documentclass[conference]{IEEEtran}
  ```

- **计算机学会期刊文章**

  ```latex
  \documentclass[10pt,journal,compsoc]{IEEEtran}
  ```

- **计算机学会会议论文**

  ```latex
  \documentclass[conference,compsoc]{IEEEtran}
  ```

- **通信学会期刊文章**

  ```latex
  \documentclass[journal,comsoc]{IEEEtran}
  ```

- **简讯、通信稿、技术短文**

  ```latex
  \documentclass[9pt,technote]{IEEEtran}
  ```

除了这些基本样式，还有其他针对投稿审稿、特殊需求的附加选项。如果你需要投稿初稿进行同行评审，IEEE 建议你一开始就采用双栏格式，以确保所有公式、表格和图形都能适配最终排版样式。

如需了解更详细的设置建议，请参考官方文档 `IEEEtran_HOWTO.pdf`。

## 第7节：如何创建常见的文章前置信息（Front Matter）

本节介绍如何编写文章的常见前置部分内容。请注意，计算机学会（Computer Society）出版物和部分会议可能有其专属的变体，本文会特别注明。

### 7.1 论文标题

使用如下方式设置标题：

```latex
\title{The Title of Your Paper}
```

请尽量避免在标题中使用数学或化学公式。

------

### 7.2 作者姓名与机构信息

作者信息可按照如下格式编写：

```latex
\author{Masahito Hayashi 
\IEEEmembership{Fellow, IEEE}, Masaki Owari
\thanks{M. Hayashi is with Graduate School 
of Mathematics, Nagoya University, Nagoya, 
Japan}
\thanks{M. Owari is with the Faculty of 
Informatics, Shizuoka University, 
Hamamatsu, Shizuoka, Japan.}
}
```

请使用 `\IEEEmembership` 命令注明作者的 IEEE 会员身份。

关于会议和计算机学会出版物中作者的具体写法，请参阅 `IEEEtran_HOWTO.pdf`。注意，整个作者组的结束大括号应放在所有 `\thanks{}` 的最后，这样可以避免生成空白的首页。

------

### 7.3 页眉（Running Heads）

使用 `\markboth` 命令声明页眉内容。此命令有两个参数：第一个是期刊名信息，第二个是作者与文章标题。

```latex
\markboth{Journal of Quantum Electronics, 
Vol. 1, No. 1, January 2021}
{Author1, Author2, 
\MakeLowercase{\textit{(et al.)}: 
Paper Title}}
```

------

### 7.4 版权声明

对于期刊文章来说，在投稿阶段无需添加版权信息，IEEE 在生产流程中会自动生成。如果是会议论文，请参考 `IEEEtran_HOWTO.pdf` 中关于“Publication ID Marks”的说明。

------

### 7.5 摘要

摘要是出现在 `\maketitle` 命令之后的第一部分，使用如下语法：

```latex
\begin{abstract}
Text of your abstract.
\end{abstract}
```

建议不要在摘要中使用数学或化学公式。

------

### 7.6 关键词（Index Terms）

关键词有助于其他研究者检索到你的文章。不同学会可能使用各自的关键词集。可联系期刊主编（EIC）获取相应关键词列表。

```latex
\begin{IEEEkeywords}
Broad band networks, quality of service
\end{IEEEkeywords}
```

------

## 第8节：如何编写正文中的常见元素

本节介绍文章正文中常见元素的使用方式及对应代码编写方法。

### 8.1 首字下沉（Drop Cap）

正文的第一段使用“首字下沉”样式，且首词为全大写。这可以通过 `\IEEEPARstart` 命令实现：

```latex
\IEEEPARstart{T}{his} is the first paragraph 
of your paper. . .
```

------

### 8.2 章节与小节标题

章节标题使用标准的 LaTeX 命令：`\section`、`\subsection` 和 `\subsubsection`。这些命令会根据不同文章类型自动处理编号。

通常建议正文在章节标题后的首段不缩进，可使用 `\noindent` 来实现：

```latex
\noindent This paragraph follows a section heading...
```

------

### 8.3 脚注

脚注可以使用 `\footnote{}` 插入：

```latex
This is a sentence with a footnote.\footnote{This is the footnote text.}
```

脚注编号会自动递增并排版于页面底部。

------

### 8.4 图像与表格

图像与表格都应作为浮动体（float）插入。下面是图像插入示例：

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[width=2.5in]{myfigure}
  \caption{图像说明}
  \label{fig_sim}
\end{figure}
```

表格插入示例：

```latex
\begin{table}[!t]
  \renewcommand{\arraystretch}{1.3}
  \caption{表格标题}
  \label{table_example}
  \centering
  \begin{tabular}{|c|c|c|}
    \hline
    项目 & 值1 & 值2 \\
    \hline
    A & 1 & 2 \\
    B & 3 & 4 \\
    \hline
  \end{tabular}
\end{table}
```

注意使用 `\label{}` 为图表设定交叉引用标识，配合 `\ref{}` 使用，例如：`Figure~\ref{fig_sim}`。

------

### 8.5 伪代码与算法环境

使用 `algorithmic` 宏包可编写伪代码：

```latex
\begin{algorithmic}
  \STATE 初始化变量
  \FOR{每一轮迭代}
    \STATE 更新参数
  \ENDFOR
\end{algorithmic}
```

------

### 8.6 数学公式

- 行内公式写在 `$ ... $` 中，例如：`$E=mc^2$`
- 独立公式使用 `equation` 环境：

```latex
\begin{equation}
E = mc^2
\label{eq:einstein}
\end{equation}
```

交叉引用可使用 `\eqref{eq:einstein}`。

------

### 8.7 引用与参考文献

引用文献使用 `\cite{文献标签}`：

```latex
Recent work \cite{IEEEexample:article_typical} shows that...
```

参考文献建议使用 `.bib` 文件与 BibTeX 结合：

```latex
\bibliographystyle{IEEEtran}
\bibliography{yourbibfile}
```

参考文献条目示例：

```bibtex
@article{IEEEexample:article_typical,
  author={A. Author and B. Writer},
  title={A Typical Paper},
  journal={IEEE Trans. Something},
  year={2021},
  volume={10},
  number={2},
  pages={123-130}
}
```

### 🔖 引用参考文献（Citations to the Bibliography）

引用使用 LaTeX 的 `\cite{}` 命令，它将以 IEEE 风格生成方括号编号的引用。例如：

1. 在导言区添加引用宏包：

```latex
\usepackage{cite}
```

1. 单个引用写法：

```latex
see \cite{ams}
```

将显示为：

> see [1]

1. 多个文献合并引用：

```latex
\cite{ams,oxford,lacomp}
```

将自动合并为类似 `[1]–[3]` 的形式。

------

### 🖼 插入图像（Figures）

图像使用标准 LaTeX 语法插入：

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=2.5in]{fig1}
\caption{这是一张图的标题}
\label{fig1}
\end{figure}
```

- `[!t]` 表示图像浮动在页面顶部，符合 IEEE 风格。
- 在导言区需要包含宏包：

```latex
\usepackage{graphicx}
```

引用图像可使用：

```latex
See Figure~\ref{fig1}
```

------

### 📊 插入表格（Tables）

表格使用 `table` 和 `tabular` 环境，示例：

```latex
\begin{table}[!t]
\begin{center}
\caption{滤波器设计公式}
\label{tab1}
\begin{tabular}{|c|c|c|}
\hline
阶数 & 任意系数 $e_m$ & 系数 $b_{ij}$ \\
\hline
1 & $b_{ij}=\hat{e}\cdot\hat{\beta_{ij}}$ & $b_{00}=0$ \\
\hline
2 & $\beta_{22}=(1,-1,-1,1,1,1)$ &  \\
\hline
3 & $b_{ij}=\hat{e}\cdot\hat{\beta_{ij}}$ & $b_{00}=0$ \\
\hline
\end{tabular}
\end{center}
\end{table}
```

引用表格时使用：

```latex
See Table~\ref{tab1}
```

## 📋 列表（Lists）

IEEEtran 支持三种常见的列表类型：无编号列表、编号列表和项目符号列表。IEEEtran 提供了多种增强选项以支持更复杂的排版需求，若需要更复杂的功能，请参阅官方文档 `IEEEtran_HOWTO.pdf`。

------

### ✅ 无编号列表（Unnumbered List）

示例输出：

- bare_jrnl.tex
- bare_conf.tex
- bare_jrnl_compsoc.tex
- bare_conf_compsoc.tex
- bare_jrnl_comsoc.tex

代码如下：

```latex
\begin{list}{}{}
\item{bare_jrnl.tex}
\item{bare_conf.tex}
\item{bare_jrnl_compsoc.tex}
\item{bare_conf_compsoc.tex}
\item{bare_jrnl_comsoc.tex}
\end{list}
```

------

### 🔢 编号列表（Numbered List）

示例输出：

1. bare_jrnl.tex
2. bare_conf.tex
3. bare_jrnl_compsoc.tex
4. bare_conf_compsoc.tex
5. bare_jrnl_comsoc.tex

代码如下：

```latex
\begin{enumerate}
\item{bare_jrnl.tex}
\item{bare_conf.tex}
\item{bare_jrnl_compsoc.tex}
\item{bare_conf_compsoc.tex}
\item{bare_jrnl_comsoc.tex}
\end{enumerate}
```

------

### 🔘 项目符号列表（Bulleted List）

示例输出：

- bare_jrnl.tex
- bare_conf.tex
- bare_jrnl_compsoc.tex
- bare_conf_compsoc.tex
- bare_jrnl_comsoc.tex

代码如下：

```latex
\begin{itemize}
\item{bare_jrnl.tex}
\item{bare_conf.tex}
\item{bare_jrnl_compsoc.tex}
\item{bare_conf_compsoc.tex}
\item{bare_jrnl_comsoc.tex}
\end{itemize}
```

------

## 🧩 其他元素（Other Elements）

对于其他较少使用的元素，例如：

- 算法（Algorithms）
- 定理与证明（Theorems and Proofs）
- 跨双栏浮动体（如整页宽度的表格、图像或公式）

建议参考文档 `IEEEtran_HOWTO.pdf` 中的“双栏浮动体（Double Column Floats）”部分。

## 第9节：如何创建常见的文末部分（Back Matter Elements）

本节展示常见的文末元素，包括：

- 致谢（Acknowledgments）
- 参考文献（Bibliographies）
- 附录（Appendices）
- 作者简介（Author Biographies）

------

### 🙏 致谢（Acknowledgments）

致谢应写成一个简洁的段落，位于参考文献之前，用于感谢在本研究过程中给予支持的个人或机构。

示例代码：

```latex
\section*{Acknowledgment}
The authors would like to thank...
```

------

### 📚 参考文献（Bibliography）

参考文献部分一般使用 BibTeX 管理。具体用法如下：

1. 在文档末尾添加：

```latex
\bibliographystyle{IEEEtran}
\bibliography{yourbibfile}
```

1. 使用 `.bib` 文件保存所有文献信息。
2. 编译顺序为：LaTeX → BibTeX → LaTeX → LaTeX

------

### 📎 附录（Appendices）

IEEEtran 支持带编号或不带编号的附录部分，代码如下：

- 单个附录：

```latex
\appendix
\section{Proof of the Main Theorem}
内容...
```

- 多个附录：

```latex
\appendices
\section{Proof of Theorem 1}
\section{Additional Simulation Results}
```

------

### 👤 作者简介（Author Biographies）

作者简介通常出现在文章末尾，IEEE 风格如下：

```latex
\begin{IEEEbiography}{John Doe}
简要介绍作者的学历、研究方向、主要贡献。建议不超过一段。
\end{IEEEbiography}
```

如果你希望加上照片，请确保使用 grayscale（灰度）图像，并控制在 1 英寸正方形以内。示例：

```latex
\begin{IEEEbiographynophoto}{Jane Doe}
内容无照片。
\end{IEEEbiographynophoto}
```

## 第十节：数学排版及其重要性

数学公式的排版规范是为了在数学文献中实现统一性与清晰度。这种规范能帮助读者更快理解作者表达的思想与新概念。

尽管 LaTeX 和 MathType® 等工具可以生成美观的数学排版，但若使用不当，也可能导致公式错误或难以阅读。

IEEE 鼓励作者遵循数学排版规范，以便撰写出更高质量的文章。

以下资料被用于制定排版指南：

- 《Mathematics into Type》（美国数学学会出版）
- 《The Printing of Mathematics》（牛津大学出版社）
- 《The LaTeX Companion》（Mittelbach 和 Goossens）
- 《More Math into LaTeX》（Grätzer）
- AMS Style Guide

更多示例可见：
 [IEEE Math Typesetting Guide](http://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Math-Typesetting-Guide.pdf)

------

### A. 显示公式（Display Equations）

使用 `equation` 环境可生成自动编号的公式：

```latex
\begin{equation}
\label{deqn_ex1}
x = \sum_{i=0}^{n} 2^{i} Q
\end{equation}
```

引用方式：

```latex
请参见式~(\ref{deqn_ex1})
```

------

### B. 公式编号规则

- 正文中的公式应从 `(1)` 连续编号到文末；
- 附录中可编号为 `(A1), (A2)`；
- 子编号如 `(1a), (1b)`，不要用 `(1-a)`、`(1.a)`；
- 避免使用罗马数字或章节号前缀。

------

### C. 多行公式与对齐

1. 多行无特定对齐：

```latex
\begin{multline}
第一行公式 \\
第二行公式 \\
第三行公式
\end{multline}
```

1. 多行对齐（如等号对齐）：

```latex
\begin{align}
a &= b + c \\
d &= e + f
\end{align}
```

1. 多点对齐：

```latex
\begin{align}
x &= y & X &= Y & a &= bc \\
x' &= y' & X' &= Y' & a' &= bz
\end{align}
```

------

### D. 子编号（Subnumbering）

```latex
\begin{subequations}\label{eq:2}
\begin{align}
f &= g \label{eq:2A} \\
f' &= g' \label{eq:2B} \\
\mathcal{L}f &= \mathcal{L}g \label{eq:2c}
\end{align}
\end{subequations}
```

------

### E. 矩阵格式

使用不同环境可生成多种括号样式的矩阵：

- 无括号：

```latex
\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}
```

- 圆括号 `pmatrix`：

```latex
\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
```

- 方括号 `bmatrix`：

```latex
\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
```

- 花括号 `Bmatrix`：
- 单竖线 `vmatrix`：
- 双竖线 `Vmatrix`：

------

### F. 数组（Arrays）

```latex
\left(
\begin{array}{cccc}
a+b+c & uv & x-y & 27 \\
a+b   & u+v & z   & 134
\end{array}
\right)
```

支持竖线 `|` 加列间隔或对齐方式 `c`, `l`, `r`。

------

### G. 分段函数结构（Cases）

```latex
\begin{equation*}
z_m(t) =
\begin{cases}
1, & \text{if } \beta_m(t) \\
0, & \text{otherwise}
\end{cases}
\end{equation*}
```

使用 `cases` 环境，可自动格式化条件函数，避免手动 `\left\{` 等繁琐代码。

------

### H. 函数格式（Function Formatting）

函数如 `min`, `arg`, `log` 等应使用 `\text{}` 保持与正文一致的字体：

```latex
\begin{equation*}
d_{R}^{KM} = \underset{d_{l}^{KM}}{\text{arg min}} \{ d_{1}^{KM}, \ldots, d_{6}^{KM} \}
\end{equation*}
```

------

### I. 缩略词排版（Text Acronyms in Equations）

例如：

```latex
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
```

------

### J. 过时写法（Obsolete Coding）

不要使用：

- `eqnarray` 环境（间距差）
- `$$ ... $$`（PlainTeX 风格）

------

### K. 推荐使用的公式环境：

| 类型       | 建议写法                 |
| ---------- | ------------------------ |
| 单行无编号 | `\[...\]` 或 `equation*` |
| 单行编号   | `equation`               |
| 多行无编号 | `align*`                 |
| 多行编号   | `align`                  |

## 第十一节：LaTeX 宏包推荐（LaTeX Package Suggestions）

在你的 LaTeX 文件开头，即 `\documentclass{}` 声明之后，应列出本论文使用的宏包。

本篇文档中使用了以下宏包：

```latex
\usepackage{amsmath,amsfonts}       % 高级数学公式支持
\usepackage{algorithmic}            % 算法伪代码环境
\usepackage{array}                  % 增强数组与表格功能
\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig} % 子图支持
\usepackage{textcomp}              % 文本符号支持，如欧元符号
\usepackage{stfloats}              % 允许双栏底部浮动图表
\usepackage{url}                   % 超链接与 URL 格式处理
\usepackage{verbatim}              % 多行注释与原文块处理
\usepackage{graphicx}              % 图像插入支持
\usepackage{balance}               % 使最后一页双栏排版保持平衡
```

### 宏包功能简介：

| 宏包名称              | 功能说明                                       |
| --------------------- | ---------------------------------------------- |
| `amsmath`、`amsfonts` | 提供数学排版能力，如对齐、多行公式、希腊字母等 |
| `algorithmic`         | 用于撰写伪代码（如 IF、FOR、WHILE）算法结构    |
| `graphicx`            | 插入图片的基础宏包，支持 `\includegraphics{}`  |
| `array`               | 增强 `tabular` 表格的控制能力                  |
| `subfig`              | 插入多张子图并配合子标题排版                   |
| `url`                 | 正确显示并换行超链接地址                       |
| `verbatim`            | 原文输入区域或多行注释块                       |
| `stfloats`            | 允许表格或图像出现在页面底部（双栏布局）       |
| `balance`             | 自动平衡最后一页双栏文本对齐                   |

## 第十二节：额外建议（Additional Advice）

以下是一些使用 LaTeX 时的常见注意事项和建议，可帮助你避免排版错误和常见陷阱：

------

### ✅ 1. 使用“软引用”而非“硬编码”

请使用类似 `\eqref{Eq}` 或 `(\ref{Eq})` 的**软引用**方式来引用图表、章节或公式。不要直接写死如 `(1)`、`Table 2` 这类文字引用。

这样做的好处是：

- 你可以自由添加/删除图表或公式；
- LaTeX 会自动更新编号；
- 避免手动修改遗漏。

------

### ⚠️ 2. `subequations` 环境会推进主公式编号

即使没有实际显示编号，`subequations` 仍会推进主编号计数。如果你忘了这点，可能会出现公式编号跳号的现象（如从 (17) 跳到 (20)），让编辑误以为你跳过了内容。

------

### 📚 3. 使用 BibTeX 时，记得提交 `.bib` 文件

BibTeX 不会“自动生成参考文献”，它依赖 `.bib` 文件获取文献信息。若使用 BibTeX 生成参考文献，请务必随稿件一并提交对应的 `.bib` 文件。

------

### 🔒 4. 避免重名标签（label）

LaTeX 不具备“读心术”，如果你将同一个标签名用于一个小节 `\subsection` 和一个表格 `\table`，LaTeX 可能会错误地将 “Table I” 引用成 “Section IV-B3”。

------

### 🕓 5. 不要在计数器更新前写 `\label`

`\label` 必须出现在更新编号的命令之后。也就是说，不要将 `\label` 写在 `\caption` 前面。否则，LaTeX 会记录上一个编号，造成引用出错。

正确顺序：

```latex
\caption{Example Figure}
\label{fig:example}
```

------

### 🚫 6. 不要在 `array` 中使用 `\nonumber` 或 `\notag`

这些命令在 `array` 环境中不起作用，可能还会意外屏蔽你希望保留的公式编号。

## 第十三节：最终检查清单（A Final Checklist）

在提交你的论文之前，请务必检查以下事项，确保你的文稿符合 IEEE 的格式规范和技术要求：

------

### ✅ 1）公式编号应连续且无重复或遗漏

- 确保所有公式从 `(1)` 开始按顺序编号；
- 避免使用连字符 `-` 或句点 `.`，应使用 `(1a)、(1b)` 这样的子编号格式；
- 若在附录中编号，使用 `(A1)、(A2)` 等格式。

------

### ✅ 2）公式格式是否规范？

- 检查文本字体、函数排版是否正确；
- `cases` 和 `array` 中的对齐是否整齐；
- 是否正确使用了 `\text{}` 来嵌入普通文字；
- 是否在多行公式中使用了合适的对齐环境，如 `align`、`multline` 等。

------

### ✅ 3）图像是否完整插入？

- 所有图像是否显示正常？
- 是否包含在编译目录下？
- 是否嵌入到 PDF 中？

------

### ✅ 4）参考文献是否完整？

- 检查引用是否对应上文中的 `\cite{}` 标签；
- 若使用 BibTeX，确保 `.bib` 文件一并提交；
- 所有文献是否按 IEEE 格式书写？

------

### 📎 附加说明：

- 请勿使用过时的 `eqnarray` 环境；
- 请勿使用 `$$ ... $$` 进行数学显示（这属于 PlainTeX，格式不规范）；
- 请确保 `\label` 命令在 `\caption` 或计数器更新之后；
- 请在表格和图像前避免使用重复标签；
- 注意不要遗漏字体嵌入，图像过低分辨率或格式错误等问题。

------

这些步骤有助于提高排版质量、减少期刊编辑返修概率，也有助于 IEEE 排版系统更高效处理你的稿件。