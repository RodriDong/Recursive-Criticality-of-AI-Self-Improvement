# Ép dùng XeLaTeX (cần cho font Unicode và tiếng Việt). Overleaf và latexmk đều đọc tệp này.
$pdf_mode = 5;
$postscript_mode = $dvi_mode = 0;
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
