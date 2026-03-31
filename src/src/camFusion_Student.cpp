#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
                         cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes;
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }
        }

        if (enclosingBoxes.size() == 1)
        {
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
    }
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            float xw = (*it2).x;
            float yw = (*it2).y;
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    float lineSpacing = 2.0;
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0);
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                             std::vector<cv::KeyPoint> &kptsPrev,
                             std::vector<cv::KeyPoint> &kptsCurr,
                             std::vector<cv::DMatch> &kptMatches)
{
    boundingBox.kptMatches.clear();
    boundingBox.keypoints.clear();

    std::vector<cv::DMatch> roiMatches;
    roiMatches.reserve(kptMatches.size());

    std::vector<double> dists;
    dists.reserve(kptMatches.size());

    for (const auto &m : kptMatches)
    {
        const cv::KeyPoint &kpCurr = kptsCurr[m.trainIdx];
        if (boundingBox.roi.contains(kpCurr.pt))
        {
            roiMatches.push_back(m);
            const cv::KeyPoint &kpPrev = kptsPrev[m.queryIdx];
            double dist = cv::norm(kpCurr.pt - kpPrev.pt);
            dists.push_back(dist);
        }
    }

    if (roiMatches.empty())
        return;

    std::vector<double> distsSorted = dists;
    std::sort(distsSorted.begin(), distsSorted.end());
    double med = distsSorted[distsSorted.size() / 2];

    std::vector<double> absDev;
    absDev.reserve(distsSorted.size());
    for (double v : distsSorted)
        absDev.push_back(std::abs(v - med));
    std::sort(absDev.begin(), absDev.end());
    double mad = absDev[absDev.size() / 2];

    double sigma = 1.4826 * mad;
    double thresh = (sigma > 1e-9) ? (med + 2.5 * sigma) : (med * 1.5);

    for (size_t i = 0; i < roiMatches.size(); ++i)
    {
        if (dists[i] <= thresh)
        {
            boundingBox.kptMatches.push_back(roiMatches[i]);
            boundingBox.keypoints.push_back(kptsCurr[roiMatches[i].trainIdx]);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches,
                      double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios;
    distRatios.reserve(kptMatches.size() * 2);

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        const cv::KeyPoint &kpOuterCurr = kptsCurr[it1->trainIdx];
        const cv::KeyPoint &kpOuterPrev = kptsPrev[it1->queryIdx];

        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
        {
            const cv::KeyPoint &kpInnerCurr = kptsCurr[it2->trainIdx];
            const cv::KeyPoint &kpInnerPrev = kptsPrev[it2->queryIdx];

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > 1e-6 && distCurr >= 1.0)
            {
                distRatios.push_back(distCurr / distPrev);
            }
        }
    }

    if (distRatios.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    double medRatio = distRatios[distRatios.size() / 2];

    double dT = 1.0 / frameRate;
    if (std::abs(1.0 - medRatio) < 1e-9)
    {
        TTC = std::numeric_limits<double>::infinity();
        return;
    }

    TTC = -dT / (1.0 - medRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate, double &TTC)
{
    const double laneHalfWidth = 2.0;

    std::vector<double> xPrev, xCurr;
    xPrev.reserve(lidarPointsPrev.size());
    xCurr.reserve(lidarPointsCurr.size());

    for (const auto &p : lidarPointsPrev)
        if (std::abs(p.y) <= laneHalfWidth)
            xPrev.push_back(p.x);

    for (const auto &p : lidarPointsCurr)
        if (std::abs(p.y) <= laneHalfWidth)
            xCurr.push_back(p.x);

    if (xPrev.empty() || xCurr.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    std::sort(xPrev.begin(), xPrev.end());
    std::sort(xCurr.begin(), xCurr.end());

    auto pickPercentile = [](const std::vector<double> &v, double p) -> double {
        size_t idx = static_cast<size_t>(p * (v.size() - 1));
        return v[idx];
    };

    double dPrev = pickPercentile(xPrev, 0.10);
    double dCurr = pickPercentile(xCurr, 0.10);

    double dT = 1.0 / frameRate;
    double denom = (dPrev - dCurr);

    if (denom <= 1e-9)
    {
        TTC = std::numeric_limits<double>::infinity();
        return;
    }

    TTC = dCurr * dT / denom;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame, DataFrame &currFrame)
{
    bbBestMatches.clear();

    std::map<std::pair<int, int>, int> pairCounts;

    for (const auto &m : matches)
    {
        const cv::KeyPoint &kpPrev = prevFrame.keypoints[m.queryIdx];
        const cv::KeyPoint &kpCurr = currFrame.keypoints[m.trainIdx];

        std::vector<int> prevIDs, currIDs;

        for (const auto &bb : prevFrame.boundingBoxes)
            if (bb.roi.contains(kpPrev.pt))
                prevIDs.push_back(bb.boxID);

        for (const auto &bb : currFrame.boundingBoxes)
            if (bb.roi.contains(kpCurr.pt))
                currIDs.push_back(bb.boxID);

        for (int pid : prevIDs)
            for (int cid : currIDs)
                pairCounts[{pid, cid}]++;
    }

    for (const auto &prevBB : prevFrame.boundingBoxes)
    {
        int bestCid = -1;
        int bestCount = 0;

        for (const auto &currBB : currFrame.boundingBoxes)
        {
            auto key = std::make_pair(prevBB.boxID, currBB.boxID);
            auto it = pairCounts.find(key);
            int c = (it != pairCounts.end()) ? it->second : 0;

            if (c > bestCount)
            {
                bestCount = c;
                bestCid = currBB.boxID;
            }
        }

        if (bestCid >= 0)
            bbBestMatches[prevBB.boxID] = bestCid;
    }
}
