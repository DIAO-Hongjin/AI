#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream> 
#include <cmath>
#include <algorithm>

using namespace std;

template< class out_type, class in_type >  
out_type convert( const in_type & value )   //��������ת�� 
{
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

struct Tree  //������ 
{
	int root_attr;  //���ڵ�����
	int leaf_label;  //Ҷ�ڵ��ǩ 
	bool const_label;  //Ҷ�ڵ���ȷ���ı�ǩ 
	vector< int > data_id;  //���Ҷ�ڵ��ϵ����� 
	vector< int > attr_value;  //���Ե�ȡֵ 
	vector< struct Tree > children;  //�ӽ�� 
	
	Tree()  //Ĭ�Ϲ��캯�� 
	{
		root_attr = -1;
		leaf_label = 0;
		const_label = false;
		data_id.clear();
		attr_value.clear();
		children.clear();
	}
};

struct Cnt  //���Լ����� 
{
	int all_cnt;  //��������Գ��ֵ��ܴ��� 
	map< int, int > label_cnt;  //�����������ĳ��ǩ���ֵĴ��� 
	
	void clear()  //���� 
	{
		all_cnt = 0;
		label_cnt.clear();
	}
};

vector< vector< int > > all_data;  //�������ݼ����������� 
vector< int > all_label;  //�������ݼ����������ݶ�Ӧ�ı�ǩ 
set< int > label_type;  //�������ݼ����б�ǩ������
map< int, set< int > > attr_type;  //�������ݵ��������Ե�����ȡֵ 
vector< int > data_id;  //�������ݼ����������ݶ�Ӧ��id 
vector< int > attr_id;  //�������ݼ�����������
vector< int > train_id;  //ѵ�������ݶ�Ӧ��id 
vector< int > validation_id;  //��֤�����ݶ�Ӧ��id  

void ReadALLData()  //��ȡ�������� 
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
				if( convert< int, string >( num ) > 15 )
				{
					if( convert< int, string >( num ) <= 20 )
						temp_vec.push_back( 20 );
					else if( convert< int, string >( num ) <= 25 )
						temp_vec.push_back( 25 );
					else if( convert< int, string >( num ) <= 30 )
						temp_vec.push_back( 30 );
					else if( convert< int, string >( num ) <= 35 )
						temp_vec.push_back( 35 );
					else if( convert< int, string >( num ) <= 40 )
						temp_vec.push_back( 40 );
					else if( convert< int, string >( num ) <= 45 )
						temp_vec.push_back( 45 );
					else 
						temp_vec.push_back( 50 );
				}
				else
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

void PartitionDataSet()
{
	for( int i = 0; i < data_id.size(); i ++ )
	{
		if( i % 10 == 0 )
		    validation_id.push_back( i );
		else
		    train_id.push_back( i );
	}
}

map< int, map< int, struct Cnt > > GetAttrInfo( vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int temp_all_cnt = 0;
	struct Cnt temp_cnt;
	vector< int > temp_vec;
	map< int, int > temp_label_cnt;
	map< int, struct Cnt > temp_attr;
	map< int, map< int, struct Cnt > > attr_info;
	
	attr_info.clear();
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	{
		temp_attr.clear();
		for( set< int > :: iterator it = attr_type[ cur_attr_id[ i ] ].begin(); it != attr_type[ cur_attr_id[ i ] ].end(); it ++ )
		{
			temp_all_cnt = 0;
			temp_label_cnt.clear();
			temp_cnt.clear();
			for( int j = 0; j < cur_data_id.size(); j ++ )
			{
				if( all_data[ cur_data_id[ j ] ][ cur_attr_id[ i ] ] == *it )
			    {
			    	temp_all_cnt ++;
			    	for( set< int > :: iterator temp_it = label_type.begin(); temp_it != label_type.end(); temp_it ++ )
			    	    if( ( *temp_it ) == all_label[ cur_data_id[ j ] ] )
			    	        temp_label_cnt[ *temp_it ] ++;
			    }
			}
			temp_cnt.all_cnt = temp_all_cnt; 
			temp_cnt.label_cnt = temp_label_cnt;
			temp_attr[ *it ] = temp_cnt;
		}
		attr_info[ cur_attr_id[ i ] ] = temp_attr;
	}
	
	return attr_info;
}

double GetHD( vector< int > cur_data_id )
{
	double hd = 0;
	map< int, int > label_sum;
	
	label_sum.clear();
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
	for( map< int, int > :: iterator it = label_sum.begin(); it != label_sum.end(); it ++ )
	    hd -= ( ( it -> second ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ) * 1.0 / cur_data_id.size() ) ) / log( 2 ) );
	
	return hd;
}

double GetHDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double hda = 0, temp = 0;
	
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 0;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) * ( log( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) ) / log( 2 ) );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		hda += temp;
	}
	
	return hda;
}

double GetGDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double gda = GetHD( cur_data_id ) - GetHDA( cur_data_id, attr_info, current_attr );
	
	return gda;
}

double GetSplitInfoDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double sida = 0;
	
	for( map< int, struct Cnt > :: iterator it = attr_info[ current_attr ].begin(); it != attr_info[ current_attr ].end(); it ++ )
	    if( ( it -> second ).all_cnt == 0 )
	        sida -= 0;
	    else
		    sida -= ( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) / log( 2 ) );
	
	return sida;
}

double GetGRatioA( vector< int > cur_data_id, vector< int > cur_attr_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )  
{
	double gra = 0, avg_sida = 0;
	
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	    avg_sida += GetSplitInfoDA( cur_data_id, attr_info, cur_attr_id[ i ] );
	avg_sida /= 1.0 * cur_attr_id.size();
	gra = GetGDA( cur_data_id, attr_info, current_attr ) / ( avg_sida + GetSplitInfoDA( cur_data_id, attr_info, current_attr ) );
	
	return gra;
}

double GetGiniDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )   
{
	double ginida = 0, temp = 0;
	
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 1;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= pow( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ), 2 );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		ginida += temp;
	}
	
	return ginida;
}

set< int > FindLabelType( vector< int > cur_data_id )
{
	set< int > result_set;
	
	result_set.clear();
	for( int i = 0; i < cur_data_id.size(); i ++ )
		result_set.insert( all_label[ cur_data_id[ i ] ] );
	
	return result_set;
}

map< pair< pair< int, int >, int >, int > GetTF( vector< int > cur_data_id )  
{
	map< pair< pair< int, int >, int >, int > attr_frq;
	
	for( int i = 0; i < cur_data_id.size(); i ++ )  
		for( int j = 0; j < attr_id.size(); j ++ )  
			attr_frq[ pair< pair< int, int>, int >( pair< int, int>( attr_id[ j ], all_data[ cur_data_id[ i ] ][ attr_id[ j ] ] ), all_label[ cur_data_id[ i ] ] ) ] ++; 
	
	return attr_frq;
}
 
int NBModel( vector< int > cur_data_id, vector< int > test_data )  //���ݻ��ֵ��ý������ݼ��Ͳ�������Ԥ��������ݵı�ǩ
{
	double a = 1;
	double max = -10000000;
	int result = 0;
	int all_length = 0;  //������������ȡֵ������ 
	map< int, double > test_prob;
	map< int, int > label_sum; 
	map< pair< pair< int, int >, int >, int > attr_frq = GetTF( cur_data_id );  //< < < ����, ȡֵ>, ��ǩ >, Ƶ�� >
	
	label_sum.clear();
	test_prob.clear();
	//����������������ȡֵ������ 
	for( map< int, set< int > > :: iterator it = attr_type.begin(); it != attr_type.end(); it ++ )
	    all_length += ( it -> second ).size();
	//ͳ�����ݼ��и���ǩ���ֵĴ���������������������� 
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
	//�������ǩ�ĺ������ 
	for( int i = 0; i < attr_id.size(); i ++ )
	    for( map< int, int > :: iterator it = label_sum.begin(); it != label_sum.end(); it ++ )
	    	test_prob[ it -> first ] += log( ( attr_frq[ pair< pair< int, int>, int >( 
			                            pair< int, int >( attr_id[ i ], test_data[ attr_id[ i ] ] ), it -> first ) ] + a ) 
										* 1.0 / ( ( it -> second ) * attr_id.size() + a * all_length ) );
	for( map< int, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	    it -> second += log( label_sum[ it -> first ] * 1.0 / cur_data_id.size() );
	//�ҵ�����������ı�ǩ����ΪԤ��ı�ǩ 
	for( map< int, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	{
		if( max < ( it -> second ) )
	    {
	    	max = ( it -> second );
	        result = it -> first;
	    }
	}
	
	return result;  //����Ԥ��ı�ǩ 
}

int ChooseRootAttr( string way, vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int root_attr = 0;
	double max = -10000, min = 10000, temp = 0;
	map< int, map< int, struct Cnt > > attr_info = GetAttrInfo( cur_data_id, cur_attr_id );
	
	if( way == "ID3" )
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
	else if( way == "C4.5" )
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
	else if( way == "CART" )
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
	
	return root_attr;
}

void CreateTree( string way, vector< int > far_data_id, vector< int > cur_data_id, vector< int > cur_attr_id, struct Tree &tree )
{
	vector< int > new_data_id, new_attr_id;
	set< int > res_label_type = FindLabelType( cur_data_id );
	
	//���ý������ݼ�ֻʣ��һ����ǩʱ�����������ΪҶ�ӽ�㣬��ǩΪʣ�µ������ǩ
	if( res_label_type.size() == 1 )
	{
		for( set< int > :: iterator it = res_label_type.begin(); it != res_label_type.end(); it ++ )
		    tree.leaf_label = *it;
		tree.const_label = true;  //��Ǹ�Ҷ���ı�ǩΪȷ���ı�ǩ
		return ;
	}
	//���ý���ϵ����ݼ��ǿռ�ʱ�����������ΪҶ�ӽ�㣬Ҷ�������ݼ�Ϊ��������ݼ� 
	if( cur_data_id.size() == 0 )
	{
		tree.data_id = far_data_id;
		return ;
	}
	//�����Լ�Ϊ��ʱ�����������ΪҶ�ӽ�㣬Ҷ�������ݼ�Ϊ��ǰ���ݼ� 
	if( cur_attr_id.size() == 0 )
	{
		tree.data_id = cur_data_id;
		return ;
	}
	//ѵ������̫��ʱ����������ûʲô���壬���м򵥼�֦�������ݼ���������С��20��ʱ������ֹͣ���� 
	if( cur_data_id.size() <= 12 )
	{
		tree.data_id = cur_data_id;  //Ҷ�������ݼ�Ϊ��ǰ���ݼ� 
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

int PredictLabel( vector< int > data, struct Tree tree )  //�������ɵľ�����Ԥ���������data�ı�ǩ 
{
	if( tree.root_attr == -1 && tree.const_label )  //ȷ����ǩ��Ҷ�ӽ�㣺����Ҷ�ӽ��ı�ǩ
	    return tree.leaf_label;
	
	if( tree.root_attr == -1 && ! tree.const_label )//��ȷ����ǩ��Ҷ�ӽ�㣺���ݽ�����ݼ���NBԤ���ǩ 
	    return NBModel( tree.data_id, data );
	
	for( int i = 0; i < tree.attr_value.size(); i ++ )  //��Ҷ�ӽ�㣺�ݹ��ҵ���Ӧ��Ҷ�ӽ��
	    if( data[ tree.root_attr ] == tree.attr_value[ i ] )
	        return PredictLabel( data, tree.children[ i ] );
	
	return *( label_type.begin() );//������˵���������ݵ�ĳ�����Գ�����ѵ�����в����ڵ�ȡֵ������ָ����ǩ 
}

void Validation( string way, vector< int > validation_id, struct Tree tree )
{
	int cnt = 0;
	
	for( int i = 0; i < validation_id.size(); i ++ )
	{
		if( all_label[ validation_id[ i ] ] == PredictLabel( all_data[ validation_id[ i ] ], tree ) )
			cnt ++;
	}
	
	cout << way << ": ��ȷ��Ϊ" << cnt * 1.0 / validation_id.size() << endl;
}

void PredictTestDataLabel( struct Tree tree )
{
	bool flag = true;
	string line = "", num = "";
	vector< int > test_data;
	ifstream in( "F://AI/lab4_Decision_Tree/test.csv" ); 
	ofstream out( "F://AI/lab4_Decision_Tree/15352076_diaohongjin.txt" );
	
	while( getline( in, line ) ) 
	{
		test_data.clear();
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				if( convert< int, string >( num ) > 15 && flag )
				{
					if( convert< int, string >( num ) <= 20 )
						test_data.push_back( 20 );
					else if( convert< int, string >( num ) <= 25 )
						test_data.push_back( 25 );
					else if( convert< int, string >( num ) <= 30 )
						test_data.push_back( 30 );
					else if( convert< int, string >( num ) <= 35 )
						test_data.push_back( 35 );
					else if( convert< int, string >( num ) <= 40 )
						test_data.push_back( 40 );
					else if( convert< int, string >( num ) <= 45 )
						test_data.push_back( 45 );
					else 
						test_data.push_back( 50 );
					flag = false;
				}
				else
					test_data.push_back( convert< int, string >( num ) );
				num = "";
			}
			else if( i == line.size() )
			{
				num = "";
				flag = true;
			}
			else
				num += line[ i ];
		}
		out << PredictLabel( test_data, tree ) << endl;
	}
	
	in.close();
	out.close();
}

int main()
{
    struct Tree ID3_tree, C45_tree, CART_tree;
    
	ReadALLData();
	PartitionDataSet();
	
	CreateTree( "ID3", train_id, train_id, attr_id, ID3_tree );
	CreateTree( "C4.5", train_id, train_id, attr_id, C45_tree );
	CreateTree( "CART", train_id, train_id, attr_id, CART_tree );
	
	Validation( "ID3", validation_id, ID3_tree );
	Validation( "C4.5", validation_id, C45_tree );
	Validation( "CART", validation_id, CART_tree );
	
	PredictTestDataLabel( CART_tree );
	
	return 0;
}


