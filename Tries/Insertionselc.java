import java.util.*;
public class Main{
    static class Node{
        Node children[]=new Node[26];
        boolean eow=false;

        Node(){
            for(int i=0;i<26;i++){
                children[i]=null;
            }
        }
    }

    public static Node root = new Node();
    public static void insert(String word){
        Node curr=root;
        for(int i=0;i<word.length();i++){
            int idx= word.charAt(i)- 'a';
            if(curr.children[idx]==null){
                curr.children[idx]=new Node();
            }
            curr=curr.children[idx];
        }
        curr.eow=true;
    }
    public static boolean search(String key){
        Node curr=root;
        for(int i=0;i<key.length();i++){
            int idx=key.charAt(i)-'a';
            if(curr.children[idx]==null){
                return false;
            }
            curr=curr.children[idx];
        }
        return curr.eow==true;
        
    }
    //wordbreak problem
    public static boolean wordBreak(String key){
        if(key.length()==0){
            return true;
        }
        for(int i=1;i<=key.length();i++){
            if(search(key.substring(0,i)) && wordBreak(key.substring(i))){
                return true;
            }
        }
        return false;
    }
    public static void main(String arg[]){
        String words[]={"i","like","sam","samsung","mobile","ice"};
        for(int i=0;i<words.length;i++){
            insert(words[i]);
        }
        // System.out.println(search("thee"));
        // System.out.print(search("their"));
        String Key="ilikesamsung";
        System.out.print(wordBreak(Key));
        
    }
}

//prefix problem

import java.util.*;
public class Main{
    static class Node{
        Node children[]=new Node[26];
        boolean eow=false;
        int freq;

        Node(){
            for(int i=0;i<26;i++){
                children[i]=null;
            }
            freq=1;
        }
    }

    public static Node root = new Node();
    public static void insert(String word){
        Node curr=root;
        for(int i=0;i<word.length();i++){
            int idx= word.charAt(i)- 'a';
            if(curr.children[idx]==null){
                curr.children[idx]=new Node();
            }
            else{
                curr.children[idx].freq++;
            }
            curr=curr.children[idx];
        }
        curr.eow=true;
    }
    public static void findPrefix(Node root,String ans){
        if(root==null){
            return;
        }
        if(root.freq==1){
            System.out.println(ans);
            return;
        }
        for(int i=0;i<root.children.length;i++){
            if(root.children[i]!=null){
                findPrefix(root.children[i],ans+(char)(i+'a'));
            }
        }
    }
    public static void main(String arg[]){
        String words[]={"zebra","dog","duck","dove"};
        for(int i=0;i<words.length;i++){
            String suffix=words.substring(i);
            insert(suffix);
        }
        root.freq=-1;
        findPrefix(root,"");
    }
}
//implement trie
class Trie {
    static class Node{
        Node children[]=new Node[26];
        boolean eow=false;

        Node(){
            for(int i=0;i<26;i++){
                children[i]=null;
            }
        }
    }
    public Node root;
    public Trie() {
        root=new Node();    
    }
    
    public void insert(String word) {
        Node curr=root;
        for(int i=0;i<word.length();i++){
            int idx=word.charAt(i)-'a';
            if(curr.children[idx]==null){
                curr.children[idx]=new Node();
            }
            curr=curr.children[idx];
        }
        curr.eow=true;
    }
    
    public boolean search(String word) {
        Node curr=root;
        for(int i=0;i<word.length();i++){
            int idx=word.charAt(i)-'a';
            if(curr.children[idx]==null){
                return false;
            }
            curr=curr.children[idx];
        }
        return curr.eow==true;
    }
    
    public boolean startsWith(String prefix) {
        Node curr=root;
        for(int i=0;i<prefix.length();i++){
            int idx=prefix.charAt(i)-'a';
            if(curr.children[idx]==null){
                return false;
            }
            curr=curr.children[idx];
        }
        return curr!=null;
    }
}

//
//unique substring count

import java.util.*;

class Solution {
    static class Node {
        Node[] children = new Node[26];
        boolean eow = false;

        Node() {
            for (int i = 0; i < 26; i++) {
                children[i] = null;
            }
        }
    }

    public static Node root = new Node();

    public static void insert(String word) {
        Node curr = root;
        for (int i = 0; i < word.length(); i++) {
            int idx = word.charAt(i) - 'a';
            if (curr.children[idx] == null) {
                curr.children[idx] = new Node();
            }
            curr = curr.children[idx];
        }
        curr.eow = true;
    }

    public int countDistinctSubstring(Node root) {
        if (root == null) {
            return 0;
        }

        int count = 0;
        for (int i = 0; i < 26; i++) {
            if (root.children[i] != null) {
                count += countDistinctSubstring(root.children[i]);
            }
        }

        return count + 1; // include current node
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s = sc.nextLine();

        for (int i = 0; i < s.length(); i++) {
            String suffix = s.substring(i);
            insert(suffix);
        }

        Solution sol = new Solution();
        int ans = sol.countDistinctSubstring(root);
        System.out.println(ans);
    }
}
//implement a phone directory

class Solution {
    static class Node{
        Node children[]=new Node[26];
        ArrayList<String> contacts=new ArrayList<>();
        
        Node(){
            for(int i=0;i<26;i++){
                children[i]=null;
            }
        }
    }
    public static Node root;
    public static void insert(String contact){
        Node curr=root;
        for(int i=0;i<contact.length();i++){
            int idx=contact.charAt(i)-'a';
            if(curr.children[idx]==null){
                curr.children[idx]=new Node();
            }
            curr=curr.children[idx];
            curr.contacts.add(contact);
        }
    }
    static ArrayList<ArrayList<String>> displayContacts(int n, String contact[],
                                                        String s) {
        root=new Node();
        Set<String> unique=new HashSet<>(Arrays.asList(contact));
        for(String c:unique){
            insert(c);
        }
        ArrayList<ArrayList<String>> result=new ArrayList<>();
        Node curr=root;
        for(int i=0;i<s.length();i++){
            int idx=s.charAt(i)-'a';
            if(curr.children[idx]==null){
                ArrayList<String> noresult=new ArrayList<>();
                noresult.add("0");
                result.add(noresult);
                curr=new Node();
                continue;
            }
            curr=curr.children[idx];
            ArrayList<String>matched=new ArrayList<>(curr.contacts);
            //Collections.sort(matched);
            result.add(matched);
        }
        return result;
    }
}
//anagram pairing 

class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        String s;
        String n;
        List<List<String>> ans=new ArrayList<>();
        HashMap<String,Integer>map=new HashMap<>();
        int j=0;
        for(int i=0;i<strs.length;i++){
            s=strs[i];
            char[]ch=s.toCharArray();
            Arrays.sort(ch);
            n=new String(ch);
            if(!map.containsKey(n)){
                map.put(n,j);
                List<String> sb=new ArrayList<>();
                sb.add(strs[i]);
                ans.add(sb);
                j++;
            }
            else{
                int k=map.get(n);
                ans.get(k).add(strs[i]);
            }
        }
        return ans;
    }
}

// M-2   HAshMAp mein hi string,List<String> banao 
//next time do it with this 













