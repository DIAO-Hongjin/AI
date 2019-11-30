#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream> 
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstdlib>

using namespace std;

template< class out_type, class in_type >  
out_type convert( const in_type & value )   //任意类型转换 
{
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

struct Tree  //决策树 
{
	int root_attr;  //根节点属性
	int leaf_label;  //叶节点标签 
	vector< int > attr_value;  //属性的取值 
	vector< struct Tree > children;  //子结点 
	
	Tree()  //默认构造函数 
	{
		root_attr = -1;  
		leaf_label = 0;
		attr_value.clear();
		children.clear();
	}
};

struct Cnt  //属性取值计数器 
{
	int all_cnt;  //计算属性该取值出现的总次数 
	map< int, int > label_cnt;  //计算属性该取值在某标签出现的次数 
	
	void clear()  //重置 
	{
		all_cnt = 0;
		label_cnt.clear();
	}
};

vector< vector< int > > all_data;  //输入数据集的所有数据 
vector< int > all_label;  //输入数据集的所有数据对应的标签 
set< int > label_type;  //输入数据集所有标签的种类
map< int, set< int > > attr_type;  //输入数据的所有属性的所有取值 
vector< int > data_id;  //输入数据集的所有数据对应的id 
vector< int > attr_id;  //输入数据集的所有属性
vector< int > train_id;  //训练集数据对应的id 
vector< int > validation_id;  //验证集数据对应的id 

void ReadALLData()  //读取输入数据 
{
	bool flag = true;
	int temp_data_cnt = 0, temp_attr_cnt = 0;
	string line = "", num = "";
	vector< int > temp_vec;
	ifstream in( "F://AI/lab4_Decision_Tree/train.csv" ); 
	
	while( getline( in, line ) ) 
	{
		temp_vec.clear();
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				temp_vec.push_back( convert< int, string >( num ) );
				num = "";
				if( flag )
					attr_id.push_back( temp_attr_cnt ++ );
			}
			else if( i == line.size() )
			{
				all_label.push_back( convert< int, string >( num ) );
				label_type.insert( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		all_data.push_back( temp_vec ); 
		data_id.push_back( temp_data_cnt ++ );
		flag = false;
	}
	for( int i = 0; i < attr_id.size(); i ++ )
	    for( int j = 0; j < data_id.size(); j ++ )
	        attr_type[ i ].insert( all_data[ j ][ i ] );
	
	in.close();
}  

void PartitionDataSet()  //将读取的数据集划分为训练集与验证集 
{
	for( int i = 0; i < data_id.size(); i ++ )
	{
		if( i % 10 == 0 )  //第1个样本、第11个样本……划分为验证集 
		    validation_id.push_back( i );  //验证集存储的是样本在总数据集中的编号 
		else  //剩余样本划分为训练集 
		    train_id.push_back( i );  //训练集存储的是样本在总数据集中的编号 
	}
}

//计算当前数据集和属性集下的属性信息 
//返回值为 < 属性编号，<属性取值，{该取值出现的总次数，该取值在某标签出现的次数} > > 
map< int, map< int, struct Cnt > > GetAttrInfo( vector< int > cur_data_id, vector< int > cur_attr_id )  
{
	int temp_all_cnt = 0;  //某一属性某一取值出现的总次数 
	struct Cnt temp_cnt;  //存储属性某个取值出现的总次数以及在某个标签出现的次数的临时变量 
	map< int, int > temp_label_cnt;  //某一属性某一取值在某个标签出现的次数 
	map< int, struct Cnt > temp_attr;  //当前数据集下某一属性的信息 
	map< int, map< int, struct Cnt > > attr_info;  //当前数据集下所有属性的信息 
	
	attr_info.clear();
	for( int i = 0; i < cur_attr_id.size(); i ++ )  //遍历当前数据集的每一种属性 
	{
		temp_attr.clear();
		//遍历该属性的每个取值
		for( set< int > :: iterator it = attr_type[ cur_attr_id[ i ] ].begin(); it != attr_type[ cur_attr_id[ i ] ].end(); it ++ )   
		{
			temp_all_cnt = 0;
			temp_label_cnt.clear();
			temp_cnt.clear();
			//遍历当前数据集以统计次数
			for( int j = 0; j < cur_data_id.size(); j ++ )   
			{
				//该属性的该取值出现次数增加  
				if( all_data[ cur_data_id[ j ] ][ cur_attr_id[ i ] ] == *it )  
			    {
			    	temp_all_cnt ++;  
			    	//遍历所有标签
			    	for( set< int > :: iterator temp_it = label_type.begin(); temp_it != label_type.end(); temp_it ++ ) 
						//当该取值在该标签的出现次数增加  
			    	    if( ( *temp_it ) == all_label[ cur_data_id[ j ] ] )   
			    	        temp_label_cnt[ *temp_it ] ++;   
			    }
			}
			temp_cnt.all_cnt = temp_all_cnt; 
			temp_cnt.label_cnt = temp_label_cnt;
			temp_attr[ *it ] = temp_cnt;  //将记录的值存入数据结构中 
		}
		attr_info[ cur_attr_id[ i ] ] = temp_attr;  //记录该属性的所有信息 
	}
	
	return attr_info;  //返回所有属性的信息 
}

double GetHD( vector< int > cur_data_id )  //计算当前数据集的经验熵 
{
	double hd = 0;
	map< int, int > label_sum;
	
	label_sum.clear();
	//计算当前数据集中各标签的出现次数 
	for( set< int > :: iterator it = label_type.begin(); it != label_type.end(); it ++ )
	{
		for( int i = 0; i < cur_data_id.size(); i ++ )
	    {
	    	if( all_label[ cur_data_id[ i ] ] == ( *it ) )
	            label_sum[ *it ] ++;
	        else
	            label_sum[ *it ] += 0;
	    }
	}
	//根据公式计算H(D) 
	for( map< int, int > :: iterator it = label_sum.begin(); it != label_sum.end(); it ++ )
	    hd -= ( ( it -> second ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ) * 1.0 / cur_data_id.size() ) ) / log( 2 ) );
	
	return hd;  //返回数据集的经验熵 
}

//计算当前数据集在当前属性条件下的条件熵 
double GetHDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double hda = 0, temp = 0;
	
	//根据公式计算当前属性的条件熵 H(D|A)
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 0;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) * ( log( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) ) / log( 2 ) );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		hda += temp;
	}
	
	return hda;  //返回数据集在当前属性条件下的条件熵 
}

//计算当前数据集在当前特征条件下的信息增益 
double GetGDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	//g(D,A) = H(D)-H(D|A) 
	double gda = GetHD( cur_data_id ) - GetHDA( cur_data_id, attr_info, current_attr );
	
	return gda;  //返回数据集在当前特征条件下的信息增益 
}


//计算当前数据集当前属性的熵 
double GetSplitInfoDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double sida = 0;
	
	for( map< int, struct Cnt > :: iterator it = attr_info[ current_attr ].begin(); it != attr_info[ current_attr ].end(); it ++ )
	    //该属性的该取值在当前数据集中出现次数为0，则标记为0，以免出现对数运算无意义的情况 
	    if( ( it -> second ).all_cnt == 0 )
	        sida -= 0;
	    else  //根据公式计算SplitInfo(D,A) 
		    sida -= ( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) / log( 2 ) );
	
	return sida;  //返回当前数据集该属性的熵 
}

//计算信息增益率 
double GetGRatioA( vector< int > cur_data_id, vector< int > cur_attr_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )  
{
	double gra = 0, avg_sida = 0;
	
	//计算所有属性的熵的平均值 
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	    avg_sida += GetSplitInfoDA( cur_data_id, attr_info, cur_attr_id[ i ] );
	avg_sida /= 1.0 * cur_attr_id.size();
	//计算信息增益率，分母加上所有属性的熵的平均值进行平滑，以免出现分母为0的情况 
	gra = GetGDA( cur_data_id, attr_info, current_attr ) / ( avg_sida + GetSplitInfoDA( cur_data_id, attr_info, current_attr ) );
	
	return gra;  //返回属性的信息增益率 
}

//计算基尼指数 
double GetGiniDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )   
{
	double ginida = 0, temp = 0;
	
	//根据公式计算基尼指数 
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 1;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= pow( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ), 2 );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		ginida += temp;
	}
	
	return ginida;  //返回基尼指数 
}

set< int > FindLabelType( vector< int > cur_data_id )
{
	set< int > result_set;
	
	result_set.clear();
	for( int i = 0; i < cur_data_id.size(); i ++ )
		result_set.insert( all_label[ cur_data_id[ i ] ] );
	
	return result_set;
}

//选取划分到该结点的样本中出现次数最多的标签作为叶子结点的标签 
int ChooseLabel( vector< int > cur_data_id )
{
	int max = 0, result = 0;
	map< int, int > cur_label_cnt;
	set< int > cur_label_set = FindLabelType( cur_data_id );
	
	cur_label_cnt.clear();
	//统计每个标签的出现次数 
	for( set< int > :: iterator it = cur_label_set.begin(); it != cur_label_set.end(); it ++ )
		cur_label_cnt[ *it ] ++;
	//找到出现次数最多的标签 
	for( map< int, int > :: iterator it = cur_label_cnt.begin(); it != cur_label_cnt.end(); it ++ )
	{
		if( max < ( it -> second ) )
		{
			max = it -> second;
			result = it -> first;
		}
	}
	
	return result;  //返回选择的标签 
}

//选择决策点的属性 
int ChooseRootAttr( string way, vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int root_attr = 0;
	double max = -10000, min = 10000, temp = 0;
	map< int, map< int, struct Cnt > > attr_info = GetAttrInfo( cur_data_id, cur_attr_id );
	
	if( way == "ID3" )  //ID3算法：选择信息增益最大的属性 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "C4.5" )  //C4.5算法：选择信息增益率最大的属性 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGRatioA( cur_data_id, cur_attr_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "CART" )  //CART算法：选择基尼系数最小的属性 
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGiniDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( min > temp )
		    {
	    		min = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	
	return root_attr;  //返回选择的属性 
}

void CreateTree( string way, vector< int > far_data_id, vector< int > cur_data_id, vector< int > cur_attr_id, struct Tree &tree )
{
	vector< int > new_data_id, new_attr_id;
	set< int > res_label_type = FindLabelType( cur_data_id );
	
	if( res_label_type.size() == 1 )
	{
		for( set< int > :: iterator it = res_label_type.begin(); it != res_label_type.end(); it ++ )
		    tree.leaf_label = *it;
		return ;
	}
	if( cur_data_id.size() == 0 )
	{
		tree.leaf_label = ChooseLabel( far_data_id );
		return ;
	}
	if( cur_attr_id.size() == 0 )
	{
		tree.leaf_label = ChooseLabel( cur_data_id );
		return ;
	}
	tree.root_attr = ChooseRootAttr( way, cur_data_id, cur_attr_id );
	for( set< int > :: iterator it = attr_type[ tree.root_attr ].begin(); it != attr_type[ tree.root_attr ].end(); it ++ )
	{
		new_data_id.clear();
		new_attr_id.clear();
		for( int j = 0; j < cur_attr_id.size(); j ++ )
		    if( cur_attr_id[ j ] != tree.root_attr )
		        new_attr_id.push_back( cur_attr_id[ j ] );
		for( int j = 0; j < cur_data_id.size(); j ++ )
		    if( all_data[ cur_data_id[ j ] ][ tree.root_attr ] == ( *it ) )
		        new_data_id.push_back( cur_data_id[ j ] ); 
		struct Tree new_tree;
		CreateTree( way, cur_data_id, new_data_id, new_attr_id, new_tree );
		tree.attr_value.push_back( *it );
		tree.children.push_back( new_tree );
	} 
}

int PredictLabel( vector< int > data, struct Tree tree )
{
	if( tree.root_attr == -1 )
	    return tree.leaf_label;
	
	for( int i = 0; i < tree.attr_value.size(); i ++ )
	    if( data[ tree.root_attr ] == tree.attr_value[ i ] )
	        return PredictLabel( data, tree.children[ i ] );
	
	return *( label_type.begin() );//到这里说明测试数据的某个属性出现了训练集中不存在的取值，返回指定标签
}

vector< struct Tree > CreateForest( string way, int size )  //生成随机森林 
{
	vector< struct Tree > forest;
	vector< int > radom_data_id;
	
	forest.clear();
	for( int i = 0; i < size; i ++ )  //生成size棵决策树 
	{
		struct Tree tree;
		srand( ( unsigned )time( NULL ) );
		radom_data_id.clear();
		for( int j = 0; j < train_id.size(); j ++ )  //有放回随机抽样产生随机训练集 
		    radom_data_id.push_back( train_id[ rand() % train_id.size() ] );
		CreateTree( way, radom_data_id, radom_data_id, attr_id, tree );  //创建决策树 
		forest.push_back( tree );  //将决策树加入森林中 
	}
	
	return forest;  //返回创建好的随机森林 
}

//得到最终预测标签
int PredictFinalLabel( vector< int > data, vector< struct Tree > forest )   
{
	int max = 0, label = 0;
	map< int, int > label_cnt;
	
	label_cnt.clear();
	for( int i = 0; i < forest.size(); i ++ )  //根据每一棵树预测标签，统计每个标签的数量 
	    label_cnt[ PredictLabel( data, forest[ i ] ) ] ++;
	//找到出现次数最多的标签 
	for( map< int, int > :: iterator it = label_cnt.begin(); it != label_cnt.end(); it ++ )
	{
		if( max < ( it -> second ) )
		{
			max = it -> second;
			label = it -> first;
		}
	}
	
	return label;  //返回出现最多的标签作为最终预测的标签 
}

void Validation( string way, vector< int > validation_id, vector< struct Tree > forest )
{
	int cnt = 0;
	
	for( int i = 0; i < validation_id.size(); i ++ )
	{
		if( all_label[ validation_id[ i ] ] == PredictFinalLabel( all_data[ validation_id[ i ] ], forest ) )
			cnt ++;
	}
	
	cout << way << ": 正确率为" << cnt * 1.0 / validation_id.size() << endl;
}

int main()
{
	ReadALLData();
	PartitionDataSet();
	
	for( int i = 0; i < 50; i ++ )
	{
		vector< struct Tree > ID3_forest = CreateForest( "ID3", 25 );
		vector< struct Tree > C45_forest = CreateForest( "C4.5", 25 );
		vector< struct Tree > CART_forest = CreateForest( "CART", 25 );
	
		Validation( "ID3", validation_id, ID3_forest );
		Validation( "C4.5", validation_id, C45_forest );
		Validation( "CART", validation_id, CART_forest );
		
		cout << endl;
	} 
	
	return 0;
}


