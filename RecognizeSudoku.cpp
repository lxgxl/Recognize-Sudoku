#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define PI 3.14159265

/*****************************
 * Some Important parameters *
 *****************************/

//width and height of image which has been resized
const int IMAGE_WIDTH = 1920;
const int IMAGE_HEIGHT = 1080;

//parameters used for thresholding
const int BLOCK_SIZE = 21;
const int THRESH_C = 15;

const int APPROXPOLYDP_EPSILON = 50;  //for approxPolyDP(), to reduce the impact of CURVED side

//parameters of sudoku
const int DIGIT_WIDTH = 32;
const int DIGIT_HEIGHT = 32;
const int SUDOKU_SIZE = 9;
const int N_MIN_ACTIVE_PIXELS = 32;

//digit size
const int SZ = 20;

//HOG parameters
Size win_size(20, 20);
Size cell_size(10, 10);
Size block_size(10, 10);
Size block_stride(5, 5);
int nbins = 9;
int deriv_apeture = 1;
int win_sigma = -1;
int histogram_norm_type = 0;
float L2_Hys_Threshold = 0.2;
int gamma_correction = 1;
int n_levels = 64;
bool signed_gradients = true;


/****************************
 * Class of sudoku          *
 ****************************/
class CSudoku{
public:
    CSudoku(int sudoku[9][9]){
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                this->sudoku[i][j] = sudoku[i][j];
                mask[i][j] = sudoku[i][j] ? 1 : 0;
                solved[i][j] = sudoku[i][j];
            }
        }

        isSolved = false;
        canSolve = true;
    }

    bool solve(){
        if(!canSolve){
            cout << "Sorry, sudoku has no solution!";
            return false;
        }

        if(isSolved){
            showSudoku(solved);
            return 1;
        }

        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(mask[i][j]){
                    row_set[i].insert(sudoku[i][j]);
                    col_set[j].insert(sudoku[i][j]);
                    block_set[i/3][j/3].insert(sudoku[i][j]);
                }
            }
        }

        if(dfs(0, 0)){
            showSudoku(solved);

            isSolved = true;

            return 1;
        }
        else{
            cout << "Sorry, sudoku has no solution!" << endl;
        }

        return 0;
    }

    void showOriginSudoku(){
        showSudoku(sudoku);
    }


private:
    int sudoku[9][9];
    int solved[9][9];
    set<int> row_set[9], col_set[9], block_set[3][3];
    bool mask[9][9];

    bool isSolved, canSolve;

    bool isOk(int row, int col, int num){
        int block_row = row / 3, block_col = col / 3;

        if(row_set[row].count(num) || col_set[col].count(num)
           || block_set[block_row][block_col].count(num)){
            return 0;
        }

        return 1;
    }
    
    bool dfs(int row, int col){
        if(row == 9){
            return 1;
        }
        if(col == 9){
            return dfs(row+1, 0);
        }
        if(mask[row][col]){
            return dfs(row, col+1);
        }

        int block_row = row / 3, block_col = col / 3;
        for(int num = 1; num <= 9; num++){
            if(!isOk(row, col, num)){
                continue;
            }

            solved[row][col] = num;
            row_set[row].insert(num);
            col_set[col].insert(num);
            block_set[block_row][block_col].insert(num);

            if(dfs(row, col+1)){
                return 1;
            }

            solved[row][col] = 0;
            row_set[row].erase(num);
            col_set[col].erase(num);
            block_set[block_row][block_col].erase(num);
        }

        return 0;
    }

    void showSudoku(int mat[9][9]){
        for(int i = 0; i < 9; i++){
                for(int j = 0; j < 9; j++){
                    cout << solved[i][j] << " ";
                    if((j+1)%3 == 0)
                        cout << " ";
                }
                if((i+1)%3 == 0)
                    cout << endl;
                cout << endl;
            }
    }
};


/**************************
 * Progress or functions  *
 **************************/

void showImage(string window_name, const Mat& image, int time = 0){
    namedWindow(window_name, WINDOW_NORMAL|CV_GUI_EXPANDED);
    imshow(window_name, image);
    waitKey(time);
    destroyWindow(window_name);
}

//f:get sudoku's contour 
vector<Point> getSukoduContour(const Mat& src_image){
    //threshold
    Mat thresheldImage;
    adaptiveThreshold(src_image, thresheldImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, BLOCK_SIZE, THRESH_C);

    //get contours
    vector<vector<Point> > contours;
    findContours(thresheldImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<Point> sudokuContour;

    // find a convex quadrilateral which is biggest
    double maxSizeOfQuadrilateral = 0;
    for(int i = 0; i < contours.size(); i++){
        //get the polygon surrounded contours
        vector<Point> candidate;
        approxPolyDP(contours[i], candidate, APPROXPOLYDP_EPSILON, true);

        //has 4 sides?
        if(candidate.size() != 4)
            continue;
        //is it convex?
        if(!isContourConvex(candidate))
            continue;

        double tmp_size = contourArea(candidate);
        if(tmp_size > maxSizeOfQuadrilateral){
            maxSizeOfQuadrilateral = tmp_size;
            sudokuContour = candidate;
        }
    }

    return sudokuContour;
}

//mark sudoku contour with red line and show in window
void showSukoduContour(Mat src_image, const vector<Point>& sudokuContour){
    for(int i = 0; i < 4; i++){
        line(src_image,
             sudokuContour[i%4],
             sudokuContour[(i+1)%4],
             Scalar(0, 0, 255),
             10
             );
    }

    showImage("SudokuContour", src_image, 0);
    return;
}

//func to sort
Point corner_center;
bool ptCompare(Point A, Point B){
    double a = atan2(A.y-corner_center.y, A.x - corner_center.x) < 0 ? atan2(A.y-corner_center.y, A.x - corner_center.x) + 2*PI : atan2(A.y-corner_center.y, A.x - corner_center.x);
    double b = atan2(B.y-corner_center.y, B.x - corner_center.x) < 0 ? atan2(B.y-corner_center.y, B.x - corner_center.x) + 2*PI : atan2(A.y-corner_center.y, A.x - corner_center.x);
    return a > b;
}

//sort counterclockwise
void sortCornerPoint(vector<Point>& corners){
    corner_center = Point(0, 0);
    for(int i = 0; i < 4; i++){
        corner_center += corners[i];
    }
    corner_center /= 4;

    sort(corners.begin(), corners.end(), ptCompare);
}

Mat getWarpImage(const Mat& gray_image, const vector<Point>& points1, const vector<Point>& points2){
    vector<Point2f> pts2f1, pts2f2;
    for(int i = 0; i < points1.size(); i++){
        pts2f1.push_back(Point2f(points1[i].x, points1[i].y));
        pts2f2.push_back(Point2f(points2[i].x, points2[i].y));
    }

    Mat warp_image;
    
    //calculate perspective matrix
    Mat persMatrix = getPerspectiveTransform(pts2f1, pts2f2);

    //get warp image
    warpPerspective(gray_image, warp_image, persMatrix, Size(SUDOKU_SIZE*DIGIT_WIDTH, SUDOKU_SIZE*DIGIT_HEIGHT));

    return warp_image;
}

Mat splitNumber(const Mat& srcImage, int x, int y){
    double max_radius = 0.4 * DIGIT_HEIGHT;

    Mat number_image;
    srcImage(Rect(x*DIGIT_WIDTH, y*DIGIT_HEIGHT, DIGIT_WIDTH, DIGIT_HEIGHT)).copyTo(number_image);

    Mat number_threshold;
    adaptiveThreshold(number_image, number_threshold, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, BLOCK_SIZE, THRESH_C);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(2,2));
    dilate(number_threshold, number_threshold, kernel);
    erode(number_threshold, number_threshold, kernel);

    for(int i = 0; i < number_threshold.rows; i++){
        uchar* data = number_threshold.ptr<uchar>(i);
        for(int j = 0; j < number_threshold.cols; j++){
            double dist = sqrt(pow(i-DIGIT_HEIGHT/2, 2) + pow(j-DIGIT_WIDTH/2, 2));
            if (dist > max_radius){
                data[j] = 0;
            }
        }
    }

    return number_threshold;
}

Rect findNumberBounding(const Mat& number_image){
    vector<vector<Point> > contours;
    findContours(number_image, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    Rect biggest_bound_rect;
    int max_size = 0;

    for(int i = 0; i < contours.size(); i++){
        Rect t_rect = boundingRect(contours[i]);
        int size = t_rect.width * t_rect.height;
        
        if(size > max_size){
            max_size = size;
            biggest_bound_rect = t_rect;
        }
    }

    biggest_bound_rect.x -= 1;
    biggest_bound_rect.y -= 1;
    biggest_bound_rect.width += 2;
    biggest_bound_rect.height += 2;

    return biggest_bound_rect;
}

bool getNumber(const Mat& srcImage, vector<Mat>& sudoku, int x, int y){
    Mat number_threshold = splitNumber(srcImage, x, y);
    int active_pixels = countNonZero(number_threshold);

    if(active_pixels > N_MIN_ACTIVE_PIXELS){
        Rect rect = findNumberBounding(number_threshold);
        int bx = rect.x, by = rect.y, w = rect.width, h = rect.height;
        
        if(!(DIGIT_WIDTH*0.2 <= (bx+w)/2 && (bx+w)/2 <= DIGIT_WIDTH*0.8
             && DIGIT_HEIGHT*0.2 <= (by+h)/2 && (by+h)/2 <= DIGIT_HEIGHT*0.8)){
            return false;
        }

        Mat number;
        number_threshold(rect).copyTo(number);
        resize(number, number, Size(SZ, SZ));
        threshold(number, number, 127, 255, THRESH_BINARY);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1,1));
        erode(number, number, kernel);
        sudoku.push_back(number);
        return true;
    }

    return false;
}

vector<int> predictNum(string filename, vector<Mat> numbers){
    //load trained data
    Ptr<ml::SVM> svm = ml::SVM::load(filename);

    //calculate HOG
    HOGDescriptor hog = HOGDescriptor(win_size,
                                      block_size, 
                                      block_stride, 
                                      cell_size,
                                      nbins,
                                      deriv_apeture,
                                      win_sigma,
                                      histogram_norm_type,
                                      L2_Hys_Threshold,
                                      gamma_correction,
                                      n_levels,
                                      signed_gradients);

    vector<int> predict_number;

    //generate test data
    for(int i = 0; i < numbers.size(); i++){
        vector<float> descriptors;
        Mat desc_matrix;

        hog.compute(numbers[i], descriptors);

        desc_matrix = Mat(1, descriptors.size(), CV_32FC1);
        int j = 0;
        for(vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++){
            desc_matrix.at<float>(0, j) = *iter;
            j++;
        }

        predict_number.push_back(svm->predict(desc_matrix));
    }

    return predict_number;
}

void showSudoku(int sudoku[9][9]){
    for(int i = 0; i < 9; i++){
        for(int j = 0; j < 9; j++){
            cout << sudoku[i][j] << " ";
            if((j+1)%3 == 0)
                cout << " ";
        }
        if((i+1)%3 == 0)
            cout << endl;
        cout << endl;
    }
}

int main(int argc, char** argv){
    string filename = argv[1];
    // string filename = "test2.jpg";
    Mat src_image = imread(filename, 1);
    showImage("src image", src_image, 0);
    Mat resized_image;
    resize(src_image, resized_image, Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    Mat gray_image;
    cvtColor(resized_image, gray_image, COLOR_BGR2GRAY);

    /**********************************************************************************************************************
     * For convenience.                                                                                                   *
     * In this part, we assume the sudoku's graphics is the biggest quadrilateral which sides can be a litter curved.     *
     **********************************************************************************************************************/

    // get contour of sudoku
    vector<Point> sudoku_contour = getSukoduContour(gray_image);
    // show contour of sudoku
    showSukoduContour(resized_image, sudoku_contour);

    //sort points of corner
    /* JUST LIKE IT
        B---A
        |   |
        C---D
    */
    sortCornerPoint(sudoku_contour);
    vector<Point> map_points = {Point(SUDOKU_SIZE*DIGIT_WIDTH, 0),
                                Point(0, 0),
                                Point(0, SUDOKU_SIZE*DIGIT_HEIGHT),
                                Point(SUDOKU_SIZE*DIGIT_WIDTH, SUDOKU_SIZE*DIGIT_HEIGHT)};
    Mat warp_image = getWarpImage(gray_image, sudoku_contour, map_points);
    showImage("warp image", warp_image);

    //get all numbers and save them
    vector<Mat> sudoku;
    int n_number = 0;
    vector<int> indexes_of_number;
    for(int i = 0; i < SUDOKU_SIZE; i++){
        for(int j = 0; j < SUDOKU_SIZE; j++){
            if(getNumber(warp_image, sudoku, j, i)){
                indexes_of_number.push_back(i*9+j);
                n_number += 1;
                imwrite("./numbers/"+to_string(i)+","+to_string(j)+".jpg", sudoku.back());
            }
        }
    }

    //recognize numbers
    vector<int> predict_numbers = predictNum("digits_svm_model2.yml", sudoku);
    
    //generate sudoku by recognized numbers
    int generatedSudoku[9][9] = {{0}};
    for(int i = 0; i < n_number; i++){
        int ind = indexes_of_number[i];
        generatedSudoku[ind/9][ind%9] = predict_numbers[i];
    }

    //create a sudoku object
    CSudoku cs(generatedSudoku);

    cout << "原数独\n";
    cs.showOriginSudoku();

    //use dfs algorithm to solve sudoku
    cout << "解决方案\n";
    cs.solve();

    getchar();
    return 0;
}