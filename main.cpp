/*
fin 二维数组，RGB最大值
matlab的filter2等同于arma的conv2
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>
#include <vector>
#include <algorithm>
#include <vector>
//using namespace arma;
static void Cv_mat_to_arma_mat(const cv::Mat& cv_mat_in, arma::mat& arma_mat_out)
{//convert unsigned int cv::Mat to arma::Mat<double>
	for (int r = 0; r < cv_mat_in.rows; r++)
	{
		for (int c = 0; c < cv_mat_in.cols; c++)
		{
			arma_mat_out(r, c) = cv_mat_in.data[r*cv_mat_in.cols + c] / 255.0;
		}
	}
};

template<typename T>
static void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out)
{
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
		static_cast<int>(arma_mat_in.n_rows),
		const_cast<T*>(arma_mat_in.memptr())),
		cv_mat_out);
};
//convert cv::Mat to arma::mat
//arma::mat img(cvImg.rows, cvImg.cols);//cvImg is a cv::Mat
//Cv_mat_to_arma_mat(cvImg, img);

//convert arma::mat to cv::Mat
//cv::Mat_<double> cv_img;
//Arma_mat_to_cv_mat<double>(arma_img, cv_img);

//取RGB中最大值转成灰度图arma
arma::mat cv3_maxTo_arma2(cv::Mat cvMat, double div)
{
	arma::mat armaMat(cvMat.rows, cvMat.cols);
	for (int r = 0; r < cvMat.rows; r++)
	{
		for (int c = 0; c < cvMat.cols; c++)
		{
			double maxNum = 0;
			for (int ch = 0; ch < cvMat.channels(); ch++)
			{
				double curNum = cvMat.at<cv::Vec3f>(r, c)[ch];
				if (curNum > maxNum) maxNum = curNum;
			}
			armaMat(r, c) = maxNum / div;
		}
	}
	return armaMat;
}
//opencv转arma
arma::mat cv2_maxTo_arma2(cv::Mat cvMat, double div)
{
	arma::mat armaMat(cvMat.rows, cvMat.cols);
	for (int r = 0; r < cvMat.rows; r++)
	{
		for (int c = 0; c < cvMat.cols; c++)
		{
			double num = cvMat.at<uchar>(r, c);
			armaMat(r, c) = num / div;
		}
	}
	return armaMat;
}


arma::sp_mat spdiags(arma::mat B, std::vector<int> v, int m, int n)
{
	arma::sp_mat A(m, n);
	if (m == n)
	{
		for (int i = 0; i < v.size(); i++)
		{
			int d = v[i];
			if (d <= 0) {
				for (int r = -d, c = 0; r < m&&c < n; r++, c++)
					A.at(r, c) = B.at(c, i);
			}
			else {
				for (int r = 0, c = d; r < m&&c < n; r++, c++)
					A.at(r, c) = B.at(c, i);
			}
			//std::cout << A << std::endl;
		}
	}
	else
	{
		//目前不需要
	}
	return A;
}

//cv::Mat tsmooth(cv::Mat I, double lambda = 0.01, int sigma = 3, double sharpness = 0.001)
arma::mat tsmooth(cv::Mat I, double lambda = 0.01, int sigma = 3, double sharpness = 0.001)
{
	//cv::resize(I, I, cv::Size(), 0.5, 0.5); //图像缩小0.5
	arma::mat fin = cv3_maxTo_arma2(I, 1.0);
	//fin.resize(fin.n_rows / 2, fin.n_cols / 2);


	arma::mat dt0_v = arma::diff(fin); //垂直差分
	dt0_v.insert_rows(dt0_v.n_rows - 1, fin.row(0) - fin.row(dt0_v.n_rows - 1));
	arma::mat dt0_h = arma::diff(fin, 1, 1); //水平差分
	dt0_h.insert_cols(dt0_h.n_cols - 1, fin.col(0) - fin.col(dt0_h.n_cols - 1));

	arma::mat kernel(1, sigma, arma::fill::ones); //滤波核
												  //kernel.ones(); //置全1
	arma::mat gauker_h = arma::conv2(dt0_h, kernel, "same");
	arma::mat gauker_v = arma::conv2(dt0_v, kernel.t(), "same");

	arma::mat wx = 1.0 / (arma::abs(gauker_h) % arma::abs(dt0_h) + sharpness);
	arma::mat wy = 1.0 / (arma::abs(gauker_v) % arma::abs(dt0_v) + sharpness);

	arma::mat IN = fin;
	int r = IN.n_rows;
	int c = IN.n_cols;
	int ch = 1;
	int k = r*c;

	arma::mat matCol = wx;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dx = -lambda*matCol;

	matCol = wy;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dy = -lambda*matCol;

	arma::mat tempx = wx.col(wx.n_cols - 1);
	tempx.insert_cols(1, wx.cols(0, wx.n_cols - 2));
	arma::mat tempy = wy.row(wy.n_rows - 1);
	tempy.insert_rows(1, wy.rows(0, wy.n_rows - 2));

	matCol = tempx;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dxa = -lambda * matCol;

	matCol = tempy;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dya = -lambda * matCol;

	tempx = wx.col(wx.n_cols - 1);
	tempx.insert_cols(1, arma::zeros(r, c - 1));

	tempy = wy.row(wy.n_rows - 1);
	tempy.insert_rows(1, arma::zeros(r - 1, c));

	matCol = tempx;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dxd1 = -lambda * matCol;

	matCol = tempy;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dyd1 = -lambda * matCol;

	wx.col(wx.n_cols - 1).fill(0.0);
	wy.row(wy.n_rows - 1).fill(0.0);

	matCol = wx;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dxd2 = -lambda * matCol;

	matCol = wy;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	arma::mat dyd2 = -lambda * matCol;

	dxd1.insert_cols(1, dxd2);
	std::vector<int> v;
	v.push_back(-k + r);
	v.push_back(-r);
	arma::sp_mat Ax = spdiags(dxd1, v, k, k);

	dyd1.insert_cols(1, dyd2);
	v.clear();
	v.push_back(-r + 1);
	v.push_back(-1);
	arma::sp_mat Ay = spdiags(dyd1, v, k, k);

	v.clear();
	v.push_back(0);
	arma::mat D = 1 - (dx + dy + dxa + dya);
	arma::sp_mat A = (Ax + Ay) + (Ax + Ay).t() + spdiags(D, v, k, k);

	//arma::superlu_opts settings;
	//settings.permutation = arma::superlu_opts::NATURAL;
	//settings.refine = arma::superlu_opts::REF_NONE;
	matCol = IN;
	matCol.reshape(matCol.n_rows*matCol.n_cols, 1);
	std::cout << A.n_rows << " " << A.n_cols << std::endl;
	//std::cout << matCol.n_rows << " " << matCol.n_cols << std::endl;
	//arma::mat outI = arma::spsolve(A, matCol); // A\IN;
	arma::mat outI = arma::spsolve(A, matCol, "superlu"); // A\IN;
														  //std::cout << "hello" << std::endl;
	outI.reshape(r, c);

	//std::cout << outI.n_rows << " " << outI.n_cols << std::endl;
	//std::cout << outI << std::endl;

	return outI;

	//cv::Mat_<double> cv_img;
	//Arma_mat_to_cv_mat<double>(outI,cv_img);
	//return cv_img;
}

arma::mat judgeBad(arma::mat t_our)
{
	arma::mat isBad(t_our.n_rows, t_our.n_cols);
	for (int i = 0; i < t_our.n_rows; i++)
	{
		for (int j = 0; j < t_our.n_cols; j++)
		{
			isBad.at(i, j) = t_our.at(i, j) < 0.5 ? 1 : 0;
		}
	}
	return isBad;
}

arma::mat rgb2gm(cv::Mat I)
{
	arma::mat ret(I.rows, I.cols);
	for (int r = 0; r < I.rows; r++)
	{
		for (int c = 0; c < I.cols; c++)
		{
			double num = 1.0;
			for (int ch = 0; ch < I.channels(); ch++) {
				num *= I.at<cv::Vec3f>(r, c)[ch];
			}
			if (num > 0) num = pow(num, 1.0 / 3);
			else num = 0;
			ret.at(r, c) = num;
		}
	}
	return ret;
}

arma::mat YisBad2(arma::mat Y, arma::mat isBad)
{
	//std::cout << Y << std::endl;
	//std::cout << isBad << std::endl;
	arma::mat ret(Y.n_rows*Y.n_cols, 1);
	int count = 0;
	for (int c = 0; c < Y.n_cols; c++)
	{
		for (int r = 0; r < Y.n_rows; r++)
		{
			//std::cout << isBad.at(r, c) << std::endl;
			if (isBad.at(r, c) > 0.9999)//等于1.0
			{
				//std::cout << isBad.at(r, c) << std::endl;
				//std::cout << Y.at(r, c) << std::endl;
				ret.at(count, 0) = Y.at(r, c);
				count++;
			}
		}
	}
	std::cout << count << std::endl;
	ret = ret.rows(0, count - 1);
	return ret;
}

arma::mat YisBad(arma::mat Y, cv::Mat isBad)
{
	//std::cout << Y << std::endl;
	//std::cout << isBad << std::endl;
	arma::mat ret(Y.n_rows*Y.n_cols, 1);
	int count = 0;
	for (int c = 0; c < Y.n_cols; c++)
	{
		for (int r = 0; r < Y.n_rows; r++)
		{
			//std::cout << isBad.at(r, c) << std::endl;
			if (isBad.at<double>(r, c) > 0.999999999)//等于1.0
			{
				//std::cout << isBad.at(r, c) << std::endl;
				//std::cout << Y.at(r, c) << std::endl;
				ret.at(count, 0) = Y.at(r, c);
				count++;
			}
		}
	}
	//std::cout << count << std::endl;
	ret = ret.rows(0, count - 1);
	return ret;
}

cv::Mat cvApplyK(cv::Mat I, double k, double a = -0.3293, double b = 1.1258)
{
	double beta = exp(b*(1 - pow(k, a)));
	double gamma = pow(k, a);
	cv::Mat I_light;
	cv::pow(I, gamma, I_light);
	return I_light*beta;
}

arma::mat armaApplyK(arma::mat I, double k, double a = -0.3293, double b = 1.1258)
{
	double beta = exp(b*(1 - pow(k, a)));
	double gamma = pow(k, a);
	return arma::pow(I, gamma)*beta;
}

double entropy(arma::mat I)
{
	double sum = 0;
	for (int c = 0; c < I.n_cols; c++)
		for (int r = 0; r < I.n_rows; r++)
		{
			double p = I.at(r, c);
			if (p > 0)	sum += p * log2(p);
		}
	return -sum;
}

double fminbnd(arma::mat Y, double mink, double maxk)
{
	double optk = mink, opte = entropy(armaApplyK(Y, mink));
	for (double k = mink + 0.001; k <= maxk; k += 0.001)
	{
		double e = entropy(armaApplyK(Y, k));
		if (e > opte)
		{
			opte = e;
			optk = k;
		}
	}
	return optk;
}

cv::Mat maxEntropyEnhance(cv::Mat input, arma::mat isB)
{
	//imshow("input", input);
	//cv::waitKey(0);

	cv::Mat I;
	cv::resize(input, I, cv::Size(50, 50), (0, 0), (0, 0), cv::INTER_NEAREST);
	//std::cout << isB << std::endl;
	cv::Mat_<double> isBad;
	Arma_mat_to_cv_mat<double>(isB, isBad);
	//std::cout << isBad << std::endl;
	//isBad.set_size(50, 50); // 与matlab的resize()不一样
	cv::resize(isBad, isBad, cv::Size(50, 50), (0, 0), (0, 0), cv::INTER_NEAREST);
	//std::cout << isBad << std::endl;
	arma::mat Y = rgb2gm(I);
	Y = YisBad(Y, isBad);
	//std::cout << Y << std::endl;


	double opt_k = fminbnd(Y, 1, 7);
	cv::Mat J = cvApplyK(input, opt_k);

	//imshow("input",input);
	//cv::waitKey(0);
	//imshow("J", J);
	//cv::waitKey(0);

	return  J;
}

cv::Mat repmat3(cv::Mat_<double> tmp)
{
	cv::Mat t(tmp.rows, tmp.cols, CV_32FC3);
	std::cout << t.rows << " " << t.cols << " " << t.channels() << " " << std::endl;
	for (int r = 0; r < tmp.rows; r++)
	{
		for (int c = 0; c < tmp.cols; c++)
		{
			t.at<cv::Vec3f>(r, c)[0] =
				t.at<cv::Vec3f>(r, c)[1] =
				t.at<cv::Vec3f>(r, c)[2] = tmp.at<double>(r, c);
		}
	}
	return t;
}

cv::Mat oneSub(cv::Mat W)
{
	cv::Mat M(W.rows, W.cols, CV_32FC3);
	for (int r = 0; r < W.rows; r++)
		for (int c = 0; c < W.cols; c++)
			for (int ch = 0; ch < W.channels(); ch++)
				M.at<cv::Vec3f>(r, c)[ch] = 1.0 - W.at<cv::Vec3f>(r, c)[ch];
	return M;
}


cv::Mat CAIP(cv::Mat imageInput, int imgScale)
{
	double mu = 0.5; // ???
	double a = -0.3293, b = 1.1258; // BTF函数参数
	double lambda = 0.5; // 照度图T的参数
	int sigma = 5; // 照度图T的参数
	cv::Mat imageDouble;
	imageInput.convertTo(imageDouble, CV_32FC3, 1 / 255.0);// CV_32FC3为要转化的类型//std::cout << imageDouble.at<cv::Vec3f>(0, 0) << std::endl;
	if (imageInput.rows>imgScale || imageInput.cols>imgScale)       //**如果太大则缩小尺寸**
		cv::resize(imageDouble, imageDouble, cv::Size(imgScale, imgScale));//不要超过2048,<=1024为16G内存之内。
	arma::mat t_our = tsmooth(imageDouble, lambda, sigma);

	cv::Mat_<double> tmp;
	Arma_mat_to_cv_mat<double>(t_our, tmp);
	cv::Mat t = repmat3(tmp);
	if (imageInput.rows>imgScale || imageInput.cols>imgScale)  //**还原尺寸**
		cv::resize(t, t, cv::Size(imageInput.cols, imageInput.rows));
	std::cout << "t的尺寸" << t.rows << " " << t.cols << " " << t.channels() << " " << std::endl;

	cv::Mat W;
	cv::pow(t, 0.5, W);
	cv::Mat I;
	imageInput.convertTo(I, CV_32FC3, 1 / 255.0);
	cv::Mat I2 = I.mul(W);
	//return I2;

	//J2
	arma::mat isBad = judgeBad(t_our);
	cv::Mat J = maxEntropyEnhance(I, isBad);
	cv::Mat W_sub = oneSub(W);
	cv::Mat J2 = J.mul(W_sub);
	//std::cout << W_sub << std::endl;

	cv::Mat result = I2 + J2;
	return result;

}

int main(int argc, char * argv[]) //命令行启动 例：ps.exe jinbo1.jpg 1
{
	//cv::Mat imageInput = cv::imread("zhangfan0.jpg");
	//std::cout << imageInput.rows << " " << imageInput.cols << std::endl;
	//std::cout << imageInput.at<cv::Vec3b>(0, 0) << std::endl;
	//cv::Mat imageInput = cv::imread("D:\\vsproject\\ps\\jinbo1.jpg");
	int imgScale[3] = { 256,512,1024 };
	cv::Mat imageInput = cv::imread(argv[1]); //命令行参数 例：jinbo1.jpg
	int sc = imgScale[atoi(argv[2]) - 1]; //命令行参数 例：1
	std::cout << argv[1] << " " << sc << std::endl;
	cv::Mat imageOutput = CAIP(imageInput, sc);
	imageOutput.convertTo(imageOutput, CV_8U, 255, 0);
	cv::imwrite("out.jpg", imageOutput);
	std::cout << "HDR完成，已保存" << std::endl;
	//imshow("原图", imageInput);
	//imshow("HDR", imageOutput);
	//cv::waitKey(0);
	return 0;
}
