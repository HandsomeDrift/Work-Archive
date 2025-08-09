# BIB Latex 模板

## 公式（Equations）

在 LaTeX 中，公式既可以以内联方式书写，也可以作为陈列公式单独排列。

对于**行内公式**，请使用 `$...$` 命令。
 例如：公式 $H\psi = E\psi$ 可以通过命令 `$H \psi = E \psi$` 来实现。

对于**陈列公式**（带自动编号），可以使用 `equation` 或 `eqnarray` 环境：

$\|\tilde{X}(k)\|^2 \leq \sum_{i=1}^{p} \left\| \tilde{Y}_i(k) \right\|^2 + \sum_{j=1}^{q} \left\| \tilde{Z}_j(k) \right\|^2 \bigg/ (p + q) \tag{1}$

其中，

$D_\mu = \partial_\mu - ig\frac{\lambda^a}{2}A^a_\mu$$F^a_{\mu\nu} = \partial_\mu A^a_\nu - \partial_\nu A^a_\mu + gf^{abc}A^b_\mu A^a_\nu \tag{2}$

请注意，在 `align` 环境中，应在每一行使用 `\nonumber`，除了最后一行，以避免对不需要编号的行自动生成编号。
 `\label{}` 命令应仅用于 `align` 环境的最后一行（即未使用 `\nonumber` 的行）。

另一个公式示例：

$Y_\infty = \left(\frac{m}{\text{GeV}}\right)^{-3} \left[1 + \frac{3 \ln(m/\text{GeV})}{15} + \frac{\ln(c^2/5)}{15} \right] \tag{3}$

该类模板还支持 `\mathbb{}`、`\mathscr{}` 和 `\mathcal{}` 命令。
 例如：`\mathbb{R}`、`\mathscr{R}` 和 `\mathcal{R}` 将分别生成不同风格的字母 **R**（参见子节 A.1.1）。

## 表格（Tables）

表格可以通过标准的 `table` 和 `tabular` 环境插入。

若要在表格中插入脚注，需使用额外的 `tablenotes` 环境包裹 `tabular` 环境。脚注将直接显示在表格下方（参见表1与表2）。

```latex
\begin{table}[t]
\begin{center}
\begin{minipage}{<width>}
\caption{<表格标题>\label{<标签>}}%
\begin{tabular}{@{}llll@{}}
\toprule
列1 & 列2 & 列3 & 列4 \\
\midrule
行1 & 数据1 & 数据2 & 数据3 \\
行2 & 数据4 & 数据5$^{1}$ & 数据6 \\
行3 & 数据7 & 数据8 & 数据9$^{2}$ \\
\botrule
\end{tabular}
\begin{tablenotes}%
\item 来源：示例数据来源。
\item[$^{1}$] 表脚注1的示例。
\item[$^{2}$] 表脚注2的示例。
\end{tablenotes}
\end{minipage}
\end{center}
\end{table}
```

------

对于不能适应正文宽度的**大型表格**，应使用旋转方式进行排版。
 为此，需使用 `\begin{sidewaystable}...\end{sidewaystable}` 环境，替代常规的 `\begin{table}...\end{table}`。

------

## 表2：示例——设置为整页宽度的长表格

| 元素11     | 元素22 |
| ---------- | ------ |
| 抛射体能量 | σ计算  |
| 元素3      | 990 A  |
| 元素4      | 500 A  |

**注**：这是一条表格脚注示例，这是一条表格脚注示例，这是一条表格脚注示例，这是一条表格脚注示例，这是一条表格脚注示例。

1 表脚注1的示例。
 2 表脚注2的示例。

## 图像（Figures）

根据 LaTeX 的图像排版标准，使用 `latex` 编译时需插入 **EPS 格式** 的图像，而使用 `pdflatex` 编译时则应插入 **PDF、JPG 或 PNG 格式** 的图像。
 这是 `latex` 与 `pdflatex` 编译方式之间的主要差别之一。

插入的图像应为 **单页文档**。
 用于插图的命令可以进行统一处理。
 用于插入图像的宏包是 `graphicx`。图像可通过标准的 `figure` 环境插入，示例如下：

```latex
\begin{figure}[t]
\centering\includegraphics{<eps-file>}
\caption{<图注>}
\label{<图像标签>}
\end{figure}
```

示例文本插入于此处。

为了演示效果，我们在 `\includegraphics` 的可选参数中包含了图像宽度，请忽略此设置。

对于不能适应正文宽度的**大型图像**，应采用旋转方式插入。
 插入旋转图像时，需使用 `\begin{sidewaysfigure}...\end{sidewaysfigure}` 环境，替代常规的 `\begin{figure}...\end{figure}` 环境。



## 算法、程序代码与代码块（Algorithms, Program codes and Listings）

排版算法需使用 `algorithm`、`algorithmicx` 和 `algpseudocode` 这些宏包。
 为此，需要使用如下格式：

```latex
\begin{algorithm}
\caption{<算法标题>}\label{<算法标签>}
\begin{algorithmic}[1]
...
\end{algorithmic}
\end{algorithm}
```

在设置 `algorithm` 环境前，建议参考上述宏包的使用文档以获取更详细的说明。

若要插入程序代码，需要使用 `program` 宏包。
 使用 `\begin{program}...\end{program}` 环境来编写程序代码。

类似地，插入代码清单需使用 `listings` 宏包。
 通过 `\begin{lstlisting}...\end{lstlisting}` 环境可实现与 `verbatim` 环境类似的功能。

更多细节请参阅 `listings` 宏包的说明文档。

------

### 示例算法 1：计算 $y = x^n$

**要求前提**：$n \geq 0$ 或 $x \neq 0$
 **确保结果**：$y = x^n$

```text
1: y ⇐ 1
2: if n < 0 then
3:  X ⇐ 1/x
4:  N ⇐ −n
5: else
6:  X ⇐ x
7:  N ⇐ n
8: end if
9: while N ≠ 0 do
10: if N 是偶数 then
11:  X ⇐ X × X
12:  N ⇐ N / 2
13: else [N 是奇数]
14:  y ⇐ y × X
15:  N ⇐ N − 1
16: end if
17: end while
```

------

```pascal
begin
 { 不执行任何操作 }
end;

Write('大小写不敏感');
Write('Pascal 关键字');
```

## 交叉引用（Cross referencing）

图、表、公式和 `align` 等环境都可以通过 `\label{#标签}` 命令设置标签。
 在 `figure` 和 `table` 环境中，`\label{}` 命令应置于 `\caption{}` 命令内部或其后。

之后可通过 `\ref{#标签}` 命令对其进行引用。
 例如，图1的标签是 `\label{fig1}`，引用时使用命令 `Figure \ref{fig1}`，输出为“Figure 1”。

------

## 引用格式说明（Details on reference citations）

使用标准的数字风格 `.bst` 文件时，只支持**数字引用格式**。
 使用作者-年份风格 `.bst` 文件时，可支持**数字引用**和**作者-年份引用**两种格式。

若选择作者-年份引用，则 `\bibitem` 项应采用以下形式之一：

```latex
\bibitem[Jones et al.(1990)]{key}...
\bibitem[Jones et al.(1990)Jones, Baker, and Williams]{key}...
\bibitem[Jones et al., 1990]{key}...
\bibitem[\protect\citeauthoryear{Jones, Baker, and Williams}{Jones et al.}{1990}]{key}...
\bibitem[\protect\citeauthoryear{Jones et al.}{1990}]{key}...
\bibitem[\protect\astroncite{Jones et al.}{1990}]{key}...
\bibitem[\protect\citename{Jones et al., }1990]{key}...
\harvarditem[Jones et al.]{Jones, Baker, and Williams}{1990}{key}...
```

这些格式可以手动编写，也可以使用适当的 `.bst` 文件通过 BibTeX 自动生成。

------

### 引用命令样式

| 引用命令      | 作者-年份模式        | 数字模式          |
| ------------- | -------------------- | ----------------- |
| `\citet{key}` | Jones et al. (1990)  | Jones et al. [21] |
| `\citep{key}` | (Jones et al., 1990) | [21]              |

------

### 多项引用示例：

```latex
\citep{key1,key2} → (Jones et al., 1990; Smith, 1989) 或 [21,24]  
\citep{key1,key2} → (Jones et al., 1990, 1991) 或 [21,24]  
\citep{key1,key2} → (Jones et al., 1990a,b) 或 [21,24]
```

`\cite{key}` 在作者-年份模式下等同于 `\citet{key}`，在数字模式下等同于 `\citep{key}`。
 要强制输出完整作者列表，可使用 `\citet*{key}` 或 `\citep*{key}`，如：

```latex
\citep*{key} → (Jones, Baker, and Mark, 1990)
```

------

### 带附加说明的引用：

```latex
\citep[chap. 2]{key} → (Jones et al., 1990, chap. 2)  
\citep[e.g.,][]{key} → (e.g., Jones et al., 1990)  
\citep[see][pg. 34]{key} → (see Jones et al., 1990, pg. 34)
```

注意：标准 LaTeX 仅允许一个注释；此模板允许两个：前注和后注。

------

### 其他引用命令：

```latex
\citealt{key} → Jones et al. 1990  
\citealt*{key} → Jones, Baker, and Williams 1990  
\citealp{key} → Jones et al., 1990  
\citealp*{key} → Jones, Baker, and Williams, 1990
```

------

### 附加命令：

```latex
\citeauthor{key} → Jones et al.  
\citeauthor*{key} → Jones, Baker, and Williams  
\citeyear{key} → 1990  
\citeyearpar{key} → (1990)  
\citetext{priv. comm.} → (priv. comm.)  
\citenum{key} → 11（非上标格式）
```

注意：完整作者列表的输出依赖于所用 `.bst` 样式文件，若样式不支持，则即便强制也无法输出全名。

------

### 特例说明

对于以小写字母开头的姓氏（如 “della Robbia”）且出现在句首：

```latex
\Citet{dRob98} → Della Robbia (1998)  
\Citep{dRob98} → (Della Robbia, 1998)  
\Citeauthor{dRob98} → Della Robbia
```

------

### 引用示例：

```latex
\cite{...} → Rahman and Abdul (2019)  
\citep{...} → (Bahdanau et al., 2014; Imboden et al., 2018; Motiian et al., 2017; Murphy, 2012; Ji et al., 2012)
```

样例引用包括：

- Krizhevsky et al. (2012)
- Horvath and Raj (2018)
- Pyrkov et al. (2018)
- Wang et al. (2018)
- LeCun et al. (2015)
- Zhang et al. (2018)
- Rav̀ı et al. (2016)

## 列表（Lists）

在 LaTeX 中，列表分为三种类型：

- **编号列表**（`enumerate` 环境）
- **项目符号列表**（`itemize` 环境）
- **无编号列表**（`unlist` 环境）

在每种环境中，使用 `\item` 命令添加一个新条目。

------

### 编号列表示例：

1. 这是第一个条目
2. `enumerate` 用于创建编号列表，`itemize` 创建项目符号列表，`unnumerate` 创建无编号列表
     a. 第二层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
     b. 第二层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
       (i) 第三层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
       (ii) 第三层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
     c. 第二层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
     d. 第二层编号列表：`enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
3. `enumerate` 创建编号列表，`itemize` 创建项目符号列表，`description` 创建无编号列表
4. 编号列表继续

------

### 项目符号列表示例：

- 第一层项目符号：这是第一个条目
- 第一层项目符号：`itemize` 创建项目符号列表，`description` 创建无编号列表
   – 第二层短横线项目符号：`itemize` 创建项目符号列表，`description` 创建无编号列表
   – 第二层短横线项目符号：`itemize` 创建项目符号列表，`description` 创建无编号列表
   – 第二层短横线项目符号：`itemize` 创建项目符号列表，`description` 创建无编号列表
- 第一层项目符号：`itemize` 创建项目符号列表，`description` 创建无编号列表
- 第一层项目符号：项目符号列表继续

------

### 无编号列表示例：

示例无编号列表文本。示例无编号列表文本。示例无编号列表文本。示例无编号列表文本。
 示例无编号列表文本。
 示例无编号列表文本。示例无编号列表文本。示例无编号列表文本。
 示例无编号列表文本。示例无编号列表文本。示例无编号列表文本。

## 类定理环境示例（Examples for theorem-like environments）

排版类定理环境需要使用 `amsthm` 宏包。

本模板中预定义了三种定理样式（`theorem styles`）：

- **thmstyleone**：带编号，定理标题为粗体，正文为斜体
- **thmstyletwo**：带编号，定理标题为正常字体，正文为斜体
- **thmstylethree**：带编号，定理标题为粗体，正文为正常字体

------

### 定理 1（定理副标题）

示例定理正文。示例定理正文。示例定理正文。示例定理正文。
 示例定理正文。示例定理正文。示例定理正文。示例定理正文。示例定理正文。

Quisque ullamcorper placerat ipsum. Cras nibh. Morbi vel justo vitae lacus tincidunt ultrices.
 Lorem ipsum dolor sit amet, consectetuer adipiscing elit. In hac habitasse platea dictumst.
 Integer tempus convallis augue.

------

### 命题 2

示例命题正文。示例命题正文。示例命题正文。示例命题正文。
 示例命题正文。示例命题正文。示例命题正文。示例命题正文。示例命题正文。

Nulla malesuada porttitor diam. Donec felis erat, congue non, volutpat at, tincidunt tristique, libero.
 Vivamus viverra fermentum felis. Donec nonummy pellentesque ante.

------

### 示例 1

Phasellus adipiscing semper elit. Proin fermentum massa ac quam.
 Sed diam turpis, molestie vitae, placerat a, molestie nec, leo.
 Maecenas lacinia. Nam ipsum ligula, eleifend at, accumsan nec, suscipit a, ipsum.
 Morbi blandit ligula feugiat magna. Nunc eleifend consequat lorem.

Nulla malesuada porttitor diam. Donec felis erat, congue non, volutpat at, tincidunt tristique, libero.
 Vivamus viverra fermentum felis. Donec nonummy pellentesque ante.

------

### 备注 1

Phasellus adipiscing semper elit. Proin fermentum massa ac quam.
 Sed diam turpis, molestie vitae, placerat a, molestie nec, leo.
 Maecenas lacinia. Nam ipsum ligula, eleifend at, accumsan nec, suscipit a, ipsum.
 Morbi blandit ligula feugiat magna. Nunc eleifend consequat lorem.

Quisque ullamcorper placerat ipsum. Cras nibh. Morbi vel justo vitae lacus tincidunt ultrices.
 Lorem ipsum dolor sit amet, consectetuer adipiscing elit. In hac habitasse platea dictumst.

------

### 定义 1（定义副标题）

示例定义正文。示例定义正文。示例定义正文。
 示例定义正文。示例定义正文。示例定义正文。示例定义正文。示例定义正文。

------

除了上述样式，还可以使用 `\begin{proof}...\end{proof}` 环境。
 该环境中的“证明”标题为斜体，正文为正常字体，并在末尾自动显示一个空心方块。

------

### 证明（Proof）

证明示例正文。证明示例正文。证明示例正文。
 证明示例正文。证明示例正文。证明示例正文。证明示例正文。证明示例正文。证明示例正文。□

Nam dui ligula, fringilla a, euismod sodales, sollicitudin vel, wisi.
 Morbi auctor lorem non justo. Nam lacus libero, pretium at, lobortis vitae, ultricies et, tellus.
 Donec aliquet, tortor sed accumsan bibendum, erat ligula aliquet magna, vitae ornare odio metus a mi.

------

### 定理 1 的证明

定理证明正文示例。定理证明正文示例。定理证明正文示例。
 定理证明正文示例。定理证明正文示例。定理证明正文示例。定理证明正文示例。定理证明正文示例。定理证明正文示例。□

## 引用段落（Quote Environment）

若要创建引用段落，应使用：

```latex
\begin{quote} ... \end{quote}
```

示例引用内容如下：

> 引用文本示例。Aliquam porttitor quam a lacus。
>  Praesent vel arcu ut tortor cursus volutpat。In vitae pede quis diam bibendum placerat。
>  Fusce elementum convallis neque。Sed dolor orci，scelerisque ac，dapibus nec，ultricies ut，mi。
>  Duis nec dui quis leo sagittis commodo。

Donec congue。Maecenas urna mi，suscipit in，placerat ut，vestibulum ut，massa。
 Fusce ultrices nulla et nisl（参见图3）。
 Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas。
 Etiam ligula arcu，elementum a，venenatis quis，sollicitudin sed，metus。
 Donec nunc pede，tincidunt in，venenatis vitae，faucibus vel（参见表3）。

------

## 结论（Conclusion）

此处填写文章结论内容。

## 第一附录的节标题

Nam dui ligula，fringilla a，euismod sodales，sollicitudin vel，wisi。
 Morbi auctor lorem non justo。Nam lacus libero，pretium at，lobortis vitae，ultricies et，tellus。
 Donec aliquet，tortor sed accumsan bibendum，erat ligula aliquet magna，vitae ornare odio metus a mi。
 Morbi ac orci et nisl hendrerit mollis。Suspendisse ut massa。Cras nec ante。Pellentesque a nulla。
 Cum sociis natoque penatibus et magnis dis parturient montes，nascetur ridiculus mus。
 Aliquam tincidunt urna。Nulla ullamcorper vestibulum turpis。Pellentesque cursus luctus mauris。

------

### 第一附录的小节标题

Nam dui ligula，fringilla a，euismod sodales，sollicitudin vel，wisi。
 Morbi auctor lorem non justo。Nam lacus libero，pretium at，lobortis vitae，ultricies et，tellus。
 Donec aliquet，tortor sed accumsan bibendum，erat ligula aliquet magna，vitae ornare odio metus a mi。
 Morbi ac orci et nisl hendrerit mollis。Suspendisse ut massa。Cras nec ante。Pellentesque a nulla。
 Cum sociis natoque penatibus et magnis dis parturient montes，nascetur ridiculus mus。
 Aliquam tincidunt urna。Nulla ullamcorper vestibulum turpis。Pellentesque cursus luctus mauris。

------

### 第一附录的子小节标题

未编号图的示例：

Fusce mauris。Vestibulum luctus nibh at lectus。
 Sed bibendum，nulla a faucibus semper，leo velit ultricies tellus，ac venenatis arcu wisi vel nisl。
 Vestibulum diam。Aliquam pellentesque，augue quis sagittis posuere，turpis lacus congue quam，
 in hendrerit risus eros eget felis。

------

### 表4：一个附录中的表格示例，显示陆军、海军和空军的食品需求

| 列1标题 | 列2标题 | 列3标题 |
| ------- | ------- | ------- |
| 列1文本 | 列2文本 | 列3文本 |
| 列1文本 | 列2文本 | 列3文本 |
| 列1文本 | 列2文本 | 列3文本 |

------

## 第二附录的节标题

Fusce mauris。Vestibulum luctus nibh at lectus。
 Sed bibendum，nulla a faucibus semper，leo velit ultricies tellus，ac venenatis arcu wisi vel nisl。
 Vestibulum diam。Aliquam pellentesque，augue quis sagittis posuere，turpis lacus congue quam，
 in hendrerit risus eros eget felis。Maecenas eget erat in sapien mattis porttitor。
 Vestibulum porttitor。Nulla facilisi。Sed a turpis eu lacus commodo facilisis。
 Morbi fringilla，wisi in dignissim interdum，justo lectus sagittis dui，et vehicula libero dui cursus dui。
 Mauris tempor ligula sed lacus。Duis cursus enim ut augue。Cras ac magna。Cras nulla。

------

### 第二附录的小节标题

Sed commodo posuere pede。Mauris ut est。Ut quis purus。Sed ac odio。Sed vehicula hendrerit sem。
 Duis non odio。Morbi ut dui。Sed accumsan risus eget odio。In hac habitasse platea dictumst。
 Pellentesque non elit。Fusce sed justo eu urna porta tincidunt。Mauris felis odio，sollicitudin sed，
 volutpat a，ornare ac，erat。Morbi quis dolor。Donec pellentesque，erat ac sagittis semper，
 nunc dui lobortis purus，quis congue purus metus ultricies tellus。
 Proin et quam。Class aptent taciti sociosqu ad litora torquent per conubia nostra，per inceptos hymenaeos。
 Praesent sapien turpis，fermentum vel，eleifend faucibus，vehicula eu，lacus。

（以上段落在该节中重复出现两次）

------

### 第二附录的子小节标题

Lorem ipsum dolor sit amet，consectetuer adipiscing elit。
 Ut purus elit，vestibulum ut，placerat ac，adipiscing vitae，felis。Curabitur dictum gravida mauris。
 Nam arcu libero，nonummy eget，consectetuer id，vulputate a，magna。
 Donec vehicula augue eu neque。

------

### 附录中公式示例：

$p = \frac{\gamma^2 - (n_C - 1)H}{(n_C - 1) + H - 2\gamma} \tag{4}$$\theta = \frac{(\gamma - H)^2(\gamma - n_C - 1)^2}{(n_C - 1 + H - 2\gamma)^2} \tag{5}$

## 另一个附录节的示例

Nam dui ligula，fringilla a，euismod sodales，sollicitudin vel，wisi。
 Morbi auctor lorem non justo。Nam lacus libero，pretium at，lobortis vitae，ultricies et，tellus。
 Donec aliquet，tortor sed accumsan bibendum，erat ligula aliquet magna，vitae ornare odio metus a mi。
 Morbi ac orci et nisl hendrerit mollis。Suspendisse ut massa。Cras nec ante。Pellentesque a nulla。
 Cum sociis natoque penatibus et magnis dis parturient montes，nascetur ridiculus mus。
 Aliquam tincidunt urna。Nulla ullamcorper vestibulum turpis。Pellentesque cursus luctus mauris。

$L = i\bar{\psi} \gamma^\mu D_\mu \psi - \frac{1}{4} F^a_{\mu\nu} F^{a\mu\nu} - m\bar{\psi}\psi \tag{6}$

Nulla malesuada porttitor diam。Donec felis erat，congue non，volutpat at，tincidunt tristique，libero。
 Vivamus viverra fermentum felis。Donec nonummy pellentesque ante。Phasellus adipiscing semper elit。
 Proin fermentum massa ac quam。Sed diam turpis，molestie vitae，placerat a，molestie nec，leo。
 Maecenas lacinia。Nam ipsum ligula，eleifend at，accumsan nec，suscipit a，ipsum。
 Morbi blandit ligula feugiat magna。Nunc eleifend consequat lorem。Sed lacinia nulla vitae enim。
 Pellentesque tincidunt purus vel magna。Integer non enim。Praesent euismod nunc eu purus。
 Donec bibendum quam in tellus。Nullam cursus pulvinar lectus。Donec et mi。Nam vulputate metus eu enim。
 Vestibulum pellentesque felis eu massa。

（以上段落在该节中重复出现两次）

------

### 表5：

| 列1标题 | 列2标题 | 列3标题 |
| ------- | ------- | ------- |
| 列1文本 | 列2文本 | 列3文本 |
| 列1文本 | 列2文本 | 列3文本 |
| 列1文本 | 列2文本 | 列3文本 |

Nulla malesuada porttitor diam。Donec felis erat，congue non，volutpat at，tincidunt tristique，libero。
 Vivamus viverra fermentum felis。Donec nonummy pellentesque ante。Phasellus adipiscing semper elit。
 Proin fermentum massa ac quam。Sed diam turpis，molestie vitae，placerat a，molestie nec，leo。
 Maecenas lacinia。Nam ipsum ligula，eleifend at，accumsan nec，suscipit a，ipsum。
 Morbi blandit ligula feugiat magna。Nunc eleifend consequat lorem。Sed lacinia nulla vitae enim。
 Pellentesque tincidunt purus vel magna。Integer non enim。Praesent euismod nunc eu purus。
 Donec bibendum quam in tellus。Nullam cursus pulvinar lectus。Donec et mi。Nam vulputate metus eu enim。
 Vestibulum pellentesque felis eu massa。Donec bibendum quam in tellus。Nullam cursus pulvinar lectus。
 Donec et mi。Nam vulputate metus eu enim。Vestibulum pellentesque felis eu massa。

------

## 竞争性利益声明（Competing interests）

作者声明不存在任何利益冲突。

------

## 作者贡献声明（Author contributions statement）

必须列出所有作者的贡献，并标明姓名首字母，例如：

S.R. 与 D.A. 共同构思了实验；S.R. 进行了实验操作；S.R. 与 D.A. 分析了实验结果；S.R. 与 D.A. 撰写并审阅了稿件。

------

## 致谢（Acknowledgments）

作者感谢匿名审稿人提出的宝贵建议。
 本研究部分受美国国家科学基金（NSF: #1636933 和 #1920920）资助。

------

## 参考文献（References）

- Bahdanau D., Cho K., Bengio Y.：通过联合对齐与翻译进行神经机器翻译（arXiv:1409.0473, 2014）
- Horvath S., Raj K.：基于DNA甲基化的生物标志物与表观遗传时钟老化理论（Nature Reviews Genetics, 2018）
- Imboden M.T. 等：心肺适能与健康人群的死亡率（JACC, 2018）
- Ji S. 等：用于动作识别的三维卷积神经网络（TPAMI, 2012）
- Krizhevsky A. 等：ImageNet 图像分类（NIPS, 2012）
- LeCun Y., Bengio Y., Hinton G.：深度学习（Nature, 2015）
- Motiian S. 等：统一的深度监督领域自适应方法（ICCV, 2017）
- Murphy K.P.：概率视角下的机器学习（MIT Press, 2012）
- Pyrkov T.V. 等：基于活动记录的生物年龄与虚弱度的量化表征（Aging, 2018）
- Rahman S.A., Abdul D.：邻域质心估算生物年龄的新方法（IEEE BHI, 2019）
- Ravì D. 等：用于健康信息学的深度学习（IEEE JBHI, 2016）
- Wang Z. 等：保持身份的条件GAN实现面部老化（CVPR, 2018）
- Zhang K. 等：基于注意力LSTM的野外精细年龄估计（arXiv:1805.10445, 2018）

------

## 作者简介（Author Biography）

作者姓名：这是作者简介的示例文本。
 可选参数中的尺寸设置仅用于示例目的，实际使用中无需包含图像的宽度与高度参数。
 这是作者简介的示例文本（多次重复省略）。

