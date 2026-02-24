//Fibonacii variations

//----1---
//climbing stairs(using recurrsion)(bad apporach)
public int climbStairs(int n){
    if(n==0){
        return 1;
    }
    if(n<0){
        return 0;
    }
    return climbStairs(n-1)+climbStairs(n-2);
}

//climbing stairs(using memorization)
class Solution {
    public int climbStairs(int n) {
        int dp[]=new int[n+1];
        return CS(dp,n);
    }
    public int CS(int dp[],int n){
        if(n==0){
            return 1;
        }
        if(n<0){
            return 0;
        }
        if(dp[n]!=0){
            return dp[n];
        }
        return dp[n]=CS(dp,n-1) + CS(dp,n-2);
    }
}

//climbing stairs(using tabulation)









//frog jump (recur + memo)

class Solution {
    int minCost(int height[]){
        int n=height.length;
        int dp[]=new int[n+1];
        return mincost1(n,height,dp);
    }
    int mincost1(int n,int[] height,int dp[]) {
        if(n==1){
            return 0;
        }
        if(n==2){
            return Math.abs(height[1]-height[0]);
        }
        if(dp[n]!=0){
            return dp[n];
        }
        return dp[n]=Math.min(mincost1(n-1,height,dp) + Math.abs(height[n-1]-height[n-2]),mincost1(n-2,height,dp) + Math.abs(height[n-1]-height[n-3]));
        
    }
}
//house robber I
class Solution {
    Recursion
    public int helper(int dp[],int []nums,int i){
        if(i<0){
            return 0;
        }
        if(i==0){
            return nums[0];
        }
        //option-1
        if(dp[i]!=0){
            return dp[i];
        }
        int take=nums[i]+helper(dp,nums,i-2);
        //option-2
        int skip=helper(dp,nums,i-1);

        dp[i]= Math.max(take,skip);
        return dp[i];
    }
    public int rob(int nums[]){
        int []dp=new int[nums.length];
        return helper(dp,nums,nums.length-1);
    }
    //Tabulation 
    public int rob(int[] nums) {
        int n = nums.length;
        if(n == 0) return 0;
        if(n == 1) return nums[0];

        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);

        for(int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i] + dp[i - 2]);
        }

        return dp[n - 1];
    }
}
//(0-1 Knapsack problem)
//(memorization)

class Solution {
    static int knapsack1(int W, int val[], int wt[],int dp[][],int n) {
       if(W==0 || n==0){
           return 0;
       }
       if(dp[n][W]!=-1){
           return dp[n][W];
       }
       if(wt[n-1]<=W){
           int ans1=val[n-1]+knapsack1(W-wt[n-1],val,wt,dp,n-1);
           int ans2=knapsack1(W,val,wt,dp,n-1);
           dp[n][W]=Math.max(ans1,ans2);
           return dp[n][W];
       }
        else{
            return knapsack1(W,val,wt,dp,n-1);
        }
    }
    static int knapsack(int W,int val[],int wt[]){
        int n=val.length;
        int dp[][]=new int[n+1][W+1];
        for(int i=0;i<dp.length;i++){
            for(int j=0;j<dp[0].length;j++){
                dp[i][j]=-1;
            }
        }
        return knapsack1(W,val,wt,dp,n);
    }
}






















//0-1 knapsack (PAINTING THE WALLS)
//O(n.n)
class Solution {
    public int paintWalls(int[] cost, int[] time) {
        int n=cost.length;
        //int m=cost.length;
        int dp[][]=new int[n+1][n+1];

        for (int[] row : dp) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        dp[0][0]=0;

        for(int i=0;i<n;i++){
            for(int j=0;j<=n;j++){
                if(dp[i][j]==Integer.MAX_VALUE){
                    continue;
                }
                int pain=Math.min(n,j+1+time[i]);
                dp[i+1][pain]=Math.min(dp[i+1][pain],dp[i][j]+cost[i]);
                dp[i+1][j]=Math.min(dp[i+1][j],dp[i][j]);
            }
        }
        return dp[n][n];
    }
}
//target sum(diff ques)

class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int n=nums.length;
        int sum=0;
        for(int i=0;i<n;i++){
            sum+=nums[i];
        }
        if (Math.abs(target) > sum || (sum + target) % 2 != 0) {
            return 0;
        }
        int p=(target+sum)/2;
        int dp[][]=new int[n+1][p+1];
        dp[0][0]=1;
        for(int i=1;i<n+1;i++){
            for(int j=0;j<p+1;j++){
                int v=nums[i-1];
                if (nums[i-1] <= j){
                    dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i-1]];
                }
                else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][p];
    }
}

//subset sum problem
//gfg

class Solution {

    static Boolean isSubsetSum(int arr[], int sum) {
        int n=arr.length;
        boolean dp[][]=new boolean[n+1][sum+1];;
        for(int i=0;i<n+1;i++){
            dp[i][0]=true;
        }
        for(int j=1;j<sum+1;j++){
            dp[0][j]=false;
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<sum+1;j++){
                if(arr[i-1]<=j){
                   dp[i][j]=dp[i-1][j-arr[i-1]] || dp[i-1][j]; 
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][sum];
    }
}


//Equal sum partition 
//leetcode

class Solution {
    public boolean canPartition(int[] nums) {
        int n=nums.length;
        int sum=0;
        for(int i=0;i<n;i++){
            sum+=nums[i];
        }
        if(sum%2!=0){
            return false;
        }
        int f=sum/2;
        boolean dp[][]=new boolean[n+1][f+1];
        for(int i=0;i<n+1;i++){
            dp[i][0]=true;
        }
        for(int j=1;j<f+1;j++){
            dp[0][j]=false;
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<f+1;j++){
                if(nums[i-1]<=j){
                    dp[i][j]=dp[i-1][j-nums[i-1]] || dp[i-1][j]; 
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][f];
    }
}

//count of subset sum with given sum
//gfg
//important change in j loop

class Solution {
    public int perfectSum(int[] nums, int target) {
        int n=nums.length;
        int dp[][]=new int[n+1][target+1];;
        for(int i=0;i<n+1;i++){
            dp[i][0]=1;
        }
        for(int j=1;j<target+1;j++){
            dp[0][j]=0;
        }
        for(int i=1;i<n+1;i++){
            for(int j=0;j<target+1;j++){
                if(nums[i-1]<=j){
                   dp[i][j]=dp[i-1][j-nums[i-1]] + dp[i-1][j]; 
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][target];
    }
}

//***** Minimum subset sum diff ******
//gfg

class Solution {
    public int minDifference(int arr[]) {
        int n=arr.length;
        int sum=0;
        for (int val:arr) {
            sum+=val;
        }
        boolean [][]dp = new boolean[n+1][sum+1];
        for (int i=0;i<=n;i++) {
            dp[i][0]=true;
        }
        for (int i=1;i<=n;i++) {
            for (int j=1;j<=sum;j++) {
                if (arr[i-1]<=j) {
                    dp[i][j]=dp[i-1][j]||dp[i-1][j-arr[i-1]];
                } else {
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        int minDiff = Integer.MAX_VALUE;
        for (int i=0;i<=sum/2;i++) {
            if (dp[n][i]) {
                minDiff = Math.min(minDiff,sum-2*i);
            }
        }
        return minDiff;
    }
}

//count of number of subset with the given difference
//gfg

//keyconcept
//s1+s2=sum(arr)------1
//s1-s2=diff-------2
//from 1 and 2
//s1=(sum+d)/2
class Solution {
    int perfectsum(int arr[],int target){
        int n=arr.length;
        int dp[][]=new int[n+1][target+1];
        for(int i=0;i<n+1;i++){
            dp[i][0]=1;
        }
        for(int j=1;j<target+1;j++){
            dp[0][j]=0;
        }
        for(int i=1;i<n+1;i++){
            for(int j=0;j<target+1;j++){
                if(arr[i-1]<=j){
                   dp[i][j]=dp[i-1][j-arr[i-1]] + dp[i-1][j]; 
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][target];
    }
    int countPartitions(int[] arr, int d) {
        int sum=0;
        for(int i=0;i<arr.length;i++){
            sum+=arr[i];
        }
        if((sum+d)%2!=0){
            return 0;
        }
        int target=(sum+d)/2;
        return perfectsum(arr,target);
    }
}

//unbounded knapsack 
//Rod cutting problem 
//gfg

class Solution {
    public int cutRod(int[] price) {
        int n=price.length;
        int []length=new int[n];
        for(int i=0;i<n;i++){
            length[i]=i+1;
        }
        int dp[][]=new int[n+1][n+1];
        for(int i=0;i<n;i++){
            dp[i][0]=0;
        }
        for(int j=0;j<n;j++){
            dp[0][j]=0;
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<n+1;j++){
                if(length[i-1]<=j){
                    dp[i][j]=Math.max(price[i-1]+dp[i][j-length[i-1]],dp[i-1][j]);
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][n];
    }
}

//coin change(count ways)
//gfg

class Solution {
    public int count(int coins[], int sum) {
        int n=coins.length;
        int dp[][]=new int[n+1][sum+1];
        for(int i=0;i<n;i++){
            dp[i][0]=1;
        }
        for(int j=1;j<sum;j++){
            dp[0][j]=0;
        }
        for(int i=1;i<=n;i++){
            for(int j=0;j<=sum;j++){
                if(coins[i-1]<=j){
                    dp[i][j]=dp[i][j-coins[i-1]] + dp[i-1][j];
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][sum];
    }
}

//coin change(minimum number of coin)

class Solution {
    public int coinChange(int[] coins, int amount) {
        int n=coins.length;
        int[][]dp=new int[n+1][amount+1];
        for(int i=0;i<=n;i++){
            dp[i][0]=0;
        }
        for(int j=0;j<=amount;j++){
            dp[0][j]=Integer.MAX_VALUE-1;
        }
        for(int j=1;j<amount+1;j++){
            if(j%coins[0]==0){
                dp[1][j]=j/coins[0];
            }
            else{
                dp[1][j]=Integer.MAX_VALUE-1;
            }
        }
        for(int i=2;i<n+1;i++){
            for(int j=1;j<amount+1;j++){
                if(coins[i-1]<=j){
                    dp[i][j]=Math.min(dp[i][j-coins[i-1]]+1,dp[i-1][j]);
                }
                else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n][amount] == Integer.MAX_VALUE - 1 ? -1 : dp[n][amount];
    }
}

//--------------L C S-------------------------------------

//LCS(Longest Common Subsequence)
class Solution {
    public int LCS(char a[], char b[],int n ,int m,int dp[][]){
        if(n==0||m==0){
            return 0;
        }
        if(dp[n][m]!=-1){
            return dp[n][m];
        }
        if(a[n-1]==b[m-1]){
            dp[n][m]=1 + LCS(a,b,n-1,m-1,dp);
            return dp[n][m];
        }
        else{
            dp[n][m]=Math.max(LCS(a,b,n,m-1,dp),LCS(a,b,n-1,m,dp));
            return dp[n][m];
        }
    }
    public int longestCommonSubsequence(String text1, String text2) {
        char a[]=text1.toCharArray();
        char b[]=text2.toCharArray();
        int n=text1.length();
        int m=text2.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<=n;i++){
            for(int j=0;j<=m;j++){
                dp[i][j]=-1;
            }
        }
        return LCS(a,b,n,m,dp);
    }
}


//Longest common substring
class Solution {
    public int longestCommonSubstr(String s1, String s2) {
        char []a=s1.toCharArray();
        char []b=s2.toCharArray();
        int n=s1.length();
        int m=s2.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<=n;i++){
            for(int j=0;j<=m;j++){
                if(i==0||j==0){
                    dp[i][j]=0;
                }
            }
        }
        int max=0;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(a[i-1]==b[j-1]){
                    dp[i][j]=1+dp[i-1][j-1];
                    max=Math.max(max,dp[i][j]);
                }
                else{
                    dp[i][j]=0;
                }
            }
        }
        return max;
    }
}

//Print the Longest subsequence

class Solution {
    public List<String> allLCS(String s1, String s2) {
        List<String> ans=new ArrayList<>();
        int n=s1.length();
        int m=s2.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<=n;i++){
            for(int j=0;j<=m;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(s1.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i][j-1],dp[i-1][j]);
                }
            }
        }
        int i=n;
        int j=m;
        while(i>0 && j>0){
            if(s1.charAt(i-1)==s2.charAt(j-1)){
                ans.add(String.valueOf(s1.charAt(i-1)));
                i--;
                j--;
            }
            else{
                if(dp[i][j-1]>dp[i-1][j]){
                    j--;
                }
                else{
                    i--;
                }
            }
        }
        Collections.reverse(ans);
        return ans;
    }
}
//print all LCS
//hard

import java.util.*;
class Solution {
    public List<String> allLCS(String s1, String s2) {
        int n = s1.length();
        int m = s2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        Map<String, Set<String>> memo = new HashMap<>();
        Set<String> result = backtrack(s1, s2, n, m, dp, memo);
        
        List<String> ans = new ArrayList<>(result);
        Collections.sort(ans);
        return ans;
    }
    private Set<String> backtrack(String s1, String s2, int i, int j, int[][] dp, Map<String, Set<String>> memo) {
        String key = i + "," + j;
        if (memo.containsKey(key)) return memo.get(key);

        Set<String> res = new HashSet<>();

        if (i == 0 || j == 0) {
            res.add("");
        } else if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            Set<String> temp = backtrack(s1, s2, i - 1, j - 1, dp, memo);
            for (String str : temp) {
                res.add(str + s1.charAt(i - 1));
            }
        } else {
            if (dp[i - 1][j] >= dp[i][j - 1]) {
                res.addAll(backtrack(s1, s2, i - 1, j, dp, memo));
            }
            if (dp[i][j - 1] >= dp[i - 1][j]) {
                res.addAll(backtrack(s1, s2, i, j - 1, dp, memo));
            }
        }

        memo.put(key, res);
        return res;
    }
}


//shortest common supersequence
//medium of gfg

class Solution {
    public static int shortestCommonSupersequence(String s1, String s2) {
        int n=s1.length();
        int m=s2.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<m+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<m+1;j++){
                if(s1.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        int f=n+m;
        int ans =f-dp[n][m];
        return ans;
    }
}

//shortest common supersequence
//hard leetcode
//good que
class Solution {
    public String shortestCommonSupersequence(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                sb.append(s1.charAt(i - 1));
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                sb.append(s1.charAt(i - 1));
                i--;
            } else {
                sb.append(s2.charAt(j - 1));
                j--;
            }
        }
        while (i > 0) {
            sb.append(s1.charAt(i - 1));
            i--;
        }
        while (j > 0) {
            sb.append(s2.charAt(j - 1));
            j--;
        }

        return sb.reverse().toString();
    }
}

//Minimum number of insertion and deletion to convert string 1 to string 2
//gfg
class Solution {
    public int minOperations(String s1, String s2) {
        int n=s1.length();
        int m=s2.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<m+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<m+1;j++){
                if(s1.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        int w=n-dp[n][m];
        int u=m-dp[n][m];
        return w+u;
    }
}

//longest common palindromic subsequence
//leetcode

class Solution {
    public int longestPalindromeSubseq(String s) {
        StringBuilder sb= new StringBuilder(s);
        String s2=sb.reverse().toString();
        int n=s.length();
        int dp[][]=new int[n+1][n+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<n+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<n+1;j++){
                if(s.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[n][n];
    }
}


//Minimum number of deletion to make a string palidrome
//leetcode hard

class Solution {
    public int minInsertions(String s) {
        StringBuilder sb=new StringBuilder(s);
        String s2=sb.reverse().toString();
        int n=s.length();
        int dp[][]=new int[n+1][n+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<n+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<n+1;j++){
                if(s.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return n-dp[n][n];
    }
}


//Longest repeating subsequence
//good gfg
class Solution {
    public int LongestRepeatingSubsequence(String s) {
        int n=s.length();
        int dp[][]=new int[n+1][n+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<n+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<n+1;j++){
                if(s.charAt(i-1)==s.charAt(j-1) && i!=j){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[n][n];
    }
}

//sequence pattern matching 
//leetcode

class Solution {
    public boolean isSubsequence(String s, String t) {
        int n=s.length();
        int m=t.length();
        int dp[][]=new int[n+1][m+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<m+1;j++){
                if(i==0 || j==0){
                    dp[i][j]=0;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<m+1;j++){
                if(s.charAt(i-1)==t.charAt(j-1)){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[n][m]==n;
    }
}

//alternative solution

class Solution {
    public boolean isSubsequence(String s, String t) {
        int leftIdx = 0;
        int rightIdx = 0;
        while (leftIdx < s.length() && rightIdx < t.length()){
            if (s.charAt(leftIdx) == t.charAt(rightIdx++)){
                leftIdx++;
            }
        }
        return leftIdx == s.length();

    }
}

//Edit distance good question
//leetcode medium 

class Solution {
    public int minDistance(String word1, String word2) {
        int n=word1.length();
        int m=word2.length();
        int [][]dp=new int[n+1][m+1];
        for(int i=0;i<n+1;i++){
            for(int j=0;j<m+1;j++){
                if(i==0){
                    dp[i][j]=j;
                }
                else if(j==0){
                    dp[i][j]=i;
                }
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<m+1;j++){
                if(word1.charAt(i-1)==word2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1];
                }
                else{
                    dp[i][j]=1+Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]));
                }
            }
        }
        return dp[n][m];
    }
}

//Wildcard matching 
//leetcode Hard

class Solution {
    public boolean isMatch(String s, String p) {
        int n=s.length();
        int m=p.length();
        boolean [][]dp=new boolean[n+1][m+1];
        for(int i=1;i<n+1;i++){
            dp[i][0]=false;
        }
        dp[0][0]=true;
        for(int i=1;i<m+1;i++){
            if(p.charAt(i-1)=='*'){
                dp[0][i]=dp[0][i-1];
            }
        }
        for(int i=1;i<n+1;i++){
            for(int j=1;j<m+1;j++){
                if(s.charAt(i-1)==p.charAt(j-1) || p.charAt(j-1)=='?'){
                    dp[i][j]=dp[i-1][j-1];
                }
                else if(p.charAt(j-1)=='*'){
                    dp[i][j]=dp[i-1][j] || dp[i][j-1];
                }
                else{
                    dp[i][j]=false;
                }
            }
        }
        return dp[n][m];
    }
}

//---------------LIS--------------
//Longest Increasing Subsequence

class Solution {
    public int lengthOfLIS(int[] nums) {
        int n=nums.length;
        int []sort=Arrays.stream(nums).distinct().sorted().toArray();
        int m=sort.length;
        int[][]dp=new int[n+1][m+1];
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                if(nums[i-1]==sort[j-1]){
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[n][m];
    }
}

//print LIS
import java.util.*;

class Solution {
    public ArrayList<Integer> getLIS(int arr[]) {
        int n = arr.length;
        int[] dp = new int[n];
        int[] parent = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(parent, -1);
        int maxLen = 1, lastIndex = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                }
            }
            if (dp[i] > maxLen) {
                maxLen = dp[i];
                lastIndex = i;
            }
        }
        ArrayList<Integer> ans = new ArrayList<>();
        while (lastIndex != -1) {
            ans.add(arr[lastIndex]);
            lastIndex = parent[lastIndex];
        }
        Collections.reverse(ans);
        return ans;
    }
}


//catalan number 
//basic code

//recursion + memo
class Solution {
    public static int findCatalan(int n) {
        int dp[]=new int[n+1];      
        return catalanmem(n,dp);
    }
    public static int catalanmem(int n,int dp[]){
        if(n==0 || n==1){
            return 1;
        }
        if(dp[n]!=0){
            return dp[n];
        }
        int ans=0;
        for(int i=0;i<n;i++){
            ans+=catalanmem(i,dp)*catalanmem(n-i-1,dp);
        }
        return dp[n]=ans;
        
    }
}

//tabulation

class Solution {
    public static int findCatalan(int n) {
        int dp[]=new int[n+1];      
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            for(int j=0;j<=i-1;j++){
                dp[i]+=dp[j]*dp[i-j-1];
            }
        }
        return dp[n];
    }
}

//mountain ranges (catalan number)

class Solution {
    public static int findCatalan(int n) {
        int dp[]=new int[n+1];      
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            for(int j=0;j<=i-1;j++){
                int inside=dp[j];
                int outside=dp[i-j-1];
                dp[i]=inside*outside;
            }
        }
        return dp[n];
    }
}

//MCM
//memo
class Solution {
    static int matrixMultiplication(int arr[]) {
        int dp[][]=new int[arr.length][arr.length];

        int n=arr.length;
        for(int i=0;i<n;i++){
            Arrays.fill(dp[i],-1);
        }
        int i=1;
        int j=arr.length-1;
        return mcm(arr,i,j,dp);
    }
    public static int mcm(int arr[],int i,int j,int dp[][]){
        if(i==j){
            return 0;
        }
        if(dp[i][j]!=-1){
            return dp[i][j];
        }
        int ans=Integer.MAX_VALUE;
        for(int k=i;k<=j-1;k++){
            int finalcost=mcm(arr,i,k,dp)+mcm(arr,k+1,j,dp)+arr[i-1]*arr[k]*arr[j];
            ans=Math.min(ans,finalcost);
        }
        return dp[i][j]=ans;
    }
}

//palindrome partioning 
//tle wala case



class Solution {
    public static boolean ispalindrome(String s,int i,int j){
        if(i==j){
            return true;
        }
        if(i>j){
            return true;
        }
        while(i<j){
            if(s.charAt(i)!=s.charAt(j)){
                return false;
            }
            else{
                i++;
                j--;
            }
        }
        return true;
    }
    static int palPartition(String s) {
        int n=s.length();
        int dp[][]=new int[n+1][n+1];
        for(int i=0;i<n;i++){
            Arrays.fill(dp[i],-1);
        }
        int i=0;
        int j=n-1;
        return pal(s,i,j,dp);
    }
    public static int pal(String s,int i,int j,int dp[][]){
        if(i>=j){
            return 0;
        }
        if(dp[i][j]!=-1){
            return dp[i][j];
        }
        if(ispalindrome(s,i,j)){
            return 0;
        }
        int min=Integer.MAX_VALUE;
        int left=0,right=0;
        for(int k=i;k<j;k++){
            if(dp[i][k]!=-1){
                left=dp[i][k];
            }
            else{
                left=pal(s,i,k,dp);
                dp[i][k]=left;
            }
            if(dp[k+1][j]!=-1){
                right=dp[k+1][j];
            }
            else{
                right=pal(s,k+1,j,dp);
                dp[k+1][j]=right;
            }
            int ans=1+right+left;
            if(ans<min){
                min=ans;
            }
        }
        return dp[i][j]=min;
    }
}



//Regular expression matching

class Solution {
    public boolean isMatch(String s, String p) {
        int l1=p.length();
        int l2=s.length();

        boolean [][]dp=new boolean[l1+1][l2+1];
        dp[0][0]=true;
        for(int i=2;i<=l1;i++){
            if(p.charAt(i-1)=='*'){
                dp[i][0]=dp[i-2][0];
            }
        }
        for(int i=1;i<=l1;i++){
            for(int j=1;j<=l2;j++){
                if(s.charAt(j-1)==p.charAt(i-1) || p.charAt(i-1)=='.'){
                    dp[i][j]=dp[i-1][j-1];
                }
                else if(p.charAt(i-1)=='*'){
                    dp[i][j]=dp[i-2][j];

                    if(p.charAt(i-2)==s.charAt(j-1) || p.charAt(i-2)=='.'){
                        dp[i][j]=dp[i][j] || dp[i][j-1];
                    }
                }
                else{
                    dp[i][j]=false;
                }
            }
        }
        return dp[l1][l2];
    }
}












