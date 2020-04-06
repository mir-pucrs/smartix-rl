SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price, sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc, count(*) AS count_order FROM LINEITEM WHERE l_shipdate <= date '1994-7-17' - interval '108' day GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment FROM PART, SUPPLIER, PARTSUPP, NATION, REGION WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 30 AND p_type LIKE '%STEEL' AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND ps_supplycost = (SELECT min(ps_supplycost) FROM PARTSUPP, SUPPLIER, NATION, REGION WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA') ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100;
SELECT l_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM CUSTOMER, ORDERS, LINEITEM WHERE c_mktsegment = 'AUTOMOBILE' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < date '1995-03-13' AND l_shipdate > date '1995-03-13' GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10;
SELECT o_orderpriority, count(*) AS order_count FROM ORDERS WHERE o_orderdate >= date '1995-01-01' AND o_orderdate < date '1995-01-01' + interval '3' month AND EXISTS (SELECT * FROM LINEITEM WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue FROM CUSTOMER, ORDERS, LINEITEM, SUPPLIER, NATION, REGION WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'MIDDLE EAST' AND o_orderdate >= date '1994-01-01' AND o_orderdate < date '1994-01-01' + interval '1' year GROUP BY n_name ORDER BY revenue DESC;
SELECT sum(l_extendedprice * l_discount) AS revenue FROM LINEITEM WHERE l_shipdate >= date '1993-01-01' AND l_shipdate < date '1993-01-01' + interval '1' year AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01 AND l_quantity < 24;
SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, extract(year FROM l_shipdate) AS l_year, l_extendedprice * (1 - l_discount) AS volume FROM SUPPLIER, LINEITEM, ORDERS, CUSTOMER, NATION n1, NATION n2 WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey AND ((n1.n_name = 'JAPAN' AND n2.n_name = 'INDIA') OR (n1.n_name = 'INDIA' AND n2.n_name = 'JAPAN')) AND l_shipdate BETWEEN date '1995-01-01' AND date '1996-12-31') AS shipping GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year;
SELECT o_year, sum(CASE WHEN nation = 'INDIA' THEN volume ELSE 0 END) / sum(volume) AS mkt_share FROM (SELECT extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation FROM PART, SUPPLIER, LINEITEM, ORDERS, CUSTOMER, NATION n1, NATION n2, REGION WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey AND r_name = 'ASIA' AND s_nationkey = n2.n_nationkey AND o_orderdate BETWEEN date '1995-01-01' AND date '1996-12-31' AND p_type = 'SMALL PLATED COPPER') AS all_nations GROUP BY o_year ORDER BY o_year;
SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM PART, SUPPLIER, LINEITEM, PARTSUPP, ORDERS, NATION WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%dim%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
SELECT c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name, c_address, c_phone, c_comment FROM CUSTOMER, ORDERS, LINEITEM, NATION WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= date '1993-08-01' AND o_orderdate < date '1993-08-01' + interval '3' month AND l_returnflag = 'R' AND c_nationkey = n_nationkey GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment ORDER BY revenue DESC LIMIT 20;
SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value FROM PARTSUPP, SUPPLIER, NATION WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'MOZAMBIQUE' GROUP BY ps_partkey HAVING sum(ps_supplycost * ps_availqty) > (SELECT sum(ps_supplycost * ps_availqty) * 0.0001000000 FROM PARTSUPP, SUPPLIER, NATION WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'MOZAMBIQUE') ORDER BY value DESC;
SELECT l_shipmode, sum(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, sum(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count FROM ORDERS, LINEITEM WHERE o_orderkey = l_orderkey AND l_shipmode IN ('RAIL', 'FOB') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND l_receiptdate >= date '1997-01-01' AND l_receiptdate < date '1997-01-01' + interval '1' year GROUP BY l_shipmode ORDER BY l_shipmode;
SELECT c_count, count(*) AS custdist FROM (SELECT c_custkey, count(o_orderkey) AS c_count FROM CUSTOMER LEFT OUTER JOIN ORDERS ON c_custkey = o_custkey AND o_comment NOT LIKE '%pending%deposits%' GROUP BY c_custkey) c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC;
SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM LINEITEM, PART WHERE l_partkey = p_partkey AND l_shipdate >= date '1996-12-01' AND l_shipdate < date '1996-12-01' + interval '1' month;
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM SUPPLIER, revenue0 WHERE s_suppkey = supplier_no AND total_revenue = (SELECT max(total_revenue) FROM revenue0) ORDER BY s_suppkey;
SELECT p_brand, p_type, p_size, count(distinct ps_suppkey) AS supplier_cnt FROM PARTSUPP, PART WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#34' AND p_type NOT LIKE 'LARGE BRUSHED%' AND p_size IN (48, 19, 12, 4, 41, 7, 21, 39) AND ps_suppkey NOT IN (SELECT s_suppkey FROM SUPPLIER WHERE s_comment LIKE '%Customer%Complaints%') GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM LINEITEM, PART WHERE p_partkey = l_partkey AND p_brand = 'Brand#44' AND p_container = 'WRAP PKG' AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM LINEITEM WHERE l_partkey = p_partkey);
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) FROM CUSTOMER, ORDERS, LINEITEM WHERE o_orderkey IN (SELECT l_orderkey FROM LINEITEM GROUP BY l_orderkey HAVING sum(l_quantity) > 314) AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100;
SELECT sum(l_extendedprice* (1 - l_discount)) AS revenue FROM LINEITEM, PART WHERE (p_partkey = l_partkey AND p_brand = 'Brand#52' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 4 AND l_quantity <= 4 + 10 AND p_size BETWEEN 1 AND 5 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#11' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND l_quantity >= 18 AND l_quantity <= 18 + 10 AND p_size BETWEEN 1 AND 10 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#51' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND l_quantity >= 29 AND l_quantity <= 29 + 10 AND p_size BETWEEN 1 AND 15 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON');
SELECT s_name, s_address FROM SUPPLIER, NATION WHERE s_suppkey IN (SELECT ps_suppkey FROM PARTSUPP WHERE ps_partkey IN (SELECT p_partkey FROM PART WHERE p_name LIKE 'green%') AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM LINEITEM WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= date '1993-01-01' AND l_shipdate < date '1993-01-01' + interval '1' year)) AND s_nationkey = n_nationkey AND n_name = 'ALGERIA' ORDER BY s_name;
SELECT s_name, count(*) AS numwait FROM SUPPLIER, LINEITEM l1, ORDERS, NATION WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS (SELECT * FROM LINEITEM l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * FROM LINEITEM l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) AND s_nationkey = n_nationkey AND n_name = 'EGYPT' GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100;
SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (SELECT substring(c_phone FROM 1 for 2) AS cntrycode, c_acctbal FROM CUSTOMER WHERE substring(c_phone FROM 1 for 2) IN ('20', '40', '22', '30', '39', '42', '21') AND c_acctbal > (SELECT avg(c_acctbal) FROM CUSTOMER WHERE c_acctbal > 0.00 AND substring(c_phone FROM 1 for 2) IN ('20', '40', '22', '30', '39', '42', '21')) AND NOT EXISTS (SELECT * FROM ORDERS WHERE o_custkey = c_custkey)) AS custsale GROUP BY cntrycode ORDER BY cntrycode;