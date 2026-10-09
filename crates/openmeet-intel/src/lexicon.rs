//! Technology lexicon (~170 common technologies with canonical casing).
//!
//! Each entry carries a regular expression. Ambiguous names that are also
//! ordinary English words ("React", "Swift", "Go", "Lambda", "Vault", …) are
//! matched case-sensitively or only in unambiguous phrases.

use std::sync::LazyLock;

use regex::{Regex, RegexSet};

/// Broad family of a technology (drives follow-up templates and topics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TechKind {
    Language,
    Framework,
    Database,
    Cache,
    Queue,
    Cloud,
    Infra,
    Observability,
    Testing,
    Security,
    DataPlatform,
    Ml,
    Protocol,
    Tool,
}

/// One lexicon entry.
#[derive(Debug)]
pub(crate) struct Tech {
    pub name: &'static str,
    pub kind: TechKind,
    pattern: &'static str,
}

const fn t(name: &'static str, kind: TechKind, pattern: &'static str) -> Tech {
    Tech {
        name,
        kind,
        pattern,
    }
}

use TechKind::*;

static TECHS: &[Tech] = &[
    // Languages
    t("Python", Language, r"(?i)\bpython\b"),
    t("Java", Language, r"(?i)\bjava\b"),
    t(
        "JavaScript",
        Language,
        r"(?i)\b(?:javascript|java script|ecmascript|vanilla js)\b",
    ),
    t("TypeScript", Language, r"(?i)\btype ?script\b"),
    t(
        "Go",
        Language,
        r"(?i:\bgolang\b|\bgo lang\b)|\b(?:in|with|using|written in|rewrote it in|and) Go\b|\bGo (?:service|services|code|backend|microservices?|binary|routines?|modules?)\b",
    ),
    t("Rust", Language, r"(?i)\brust\b"),
    t("C++", Language, r"(?i)(?:^|[^\w+])c\+\+(?:[^\w+]|$)"),
    t(
        "C#",
        Language,
        r"(?i)(?:^|[^\w#])c#(?:[^\w#]|$)|\bc sharp\b",
    ),
    t("Kotlin", Language, r"(?i)\bkotlin\b"),
    t("Swift", Language, r"\bSwift\b"),
    t("Scala", Language, r"(?i)\bscala\b"),
    t("Ruby", Language, r"(?i)\bruby\b"),
    t("PHP", Language, r"(?i)\bphp\b"),
    t("Elixir", Language, r"(?i)\belixir\b"),
    t("Erlang", Language, r"(?i)\berlang\b"),
    t("Haskell", Language, r"(?i)\bhaskell\b"),
    t("Clojure", Language, r"(?i)\bclojure\b"),
    t("Dart", Language, r"\bDart\b"),
    t("SQL", Language, r"(?i)\bsql\b"),
    t("NoSQL", Database, r"(?i)\bno-?sql\b"),
    t("Bash", Language, r"(?i)\bbash\b"),
    t("PowerShell", Language, r"(?i)\bpower ?shell\b"),
    t("WebAssembly", Language, r"(?i)\b(?:webassembly|wasm)\b"),
    // Frameworks & runtimes
    t("Node.js", Framework, r"(?i)\bnode(?:\.js|js| js)\b"),
    t("Deno", Framework, r"(?i)\bdeno\b"),
    t(
        "React",
        Framework,
        r"\bReact(?:\.js|JS)?\b|(?i:\breact(?:\.js|js| app| components?| frontend| hooks?)\b)",
    ),
    t("React Native", Framework, r"(?i)\breact native\b"),
    t("Angular", Framework, r"(?i)\bangular(?:js)?\b"),
    t("Vue", Framework, r"(?i)\bvue(?:\.js|js)?\b"),
    t("Svelte", Framework, r"(?i)\bsvelte(?:kit)?\b"),
    t("Next.js", Framework, r"(?i)\bnext\.?js\b|\bnext js\b"),
    t("Nuxt", Framework, r"(?i)\bnuxt\b"),
    t(
        "Express",
        Framework,
        r"(?i:\bexpress(?:\.js|js)\b)|\bExpress\b",
    ),
    t("Django", Framework, r"(?i)\bdjango\b"),
    t("Flask", Framework, r"(?i)\bflask\b"),
    t("FastAPI", Framework, r"(?i)\bfast ?api\b"),
    t(
        "Ruby on Rails",
        Framework,
        r"(?i:\bruby on rails\b)|\bRails\b",
    ),
    t(
        "Spring Boot",
        Framework,
        r"(?i)\bspring(?: boot| framework)\b",
    ),
    t(
        ".NET",
        Framework,
        r"(?i)(?:^|[^\w.])\.net\b|\bdotnet\b|\basp\.net\b",
    ),
    t("Laravel", Framework, r"(?i)\blaravel\b"),
    t("Tauri", Framework, r"(?i)\btauri\b"),
    t("Electron", Framework, r"\bElectron\b"),
    t("Flutter", Framework, r"(?i)\bflutter\b"),
    t("SwiftUI", Framework, r"(?i)\bswift ?ui\b"),
    t("jQuery", Framework, r"(?i)\bjquery\b"),
    t("Tailwind CSS", Framework, r"(?i)\btailwind(?:css| css)?\b"),
    t("GraphQL", Protocol, r"(?i)\bgraph ?ql\b"),
    t("gRPC", Protocol, r"(?i)\bgrpc\b"),
    t(
        "REST",
        Protocol,
        r"\bREST(?:ful)?\b|(?i:\brest(?:ful)? apis?\b)",
    ),
    t("WebSockets", Protocol, r"(?i)\bweb ?sockets?\b"),
    t(
        "Protobuf",
        Protocol,
        r"(?i)\b(?:protobuf|protocol buffers)\b",
    ),
    t("OpenAPI", Tool, r"(?i)\bopen ?api\b|\bswagger\b"),
    // Databases & storage
    t("PostgreSQL", Database, r"(?i)\bpostgres(?:ql)?\b"),
    t("MySQL", Database, r"(?i)\bmy ?sql\b"),
    t("MariaDB", Database, r"(?i)\bmaria ?db\b"),
    t("SQLite", Database, r"(?i)\bsqlite\b"),
    t("MongoDB", Database, r"(?i)\bmongo(?:db)?\b"),
    t("Redis", Cache, r"(?i)\bredis\b"),
    t("Memcached", Cache, r"(?i)\bmemcached?\b"),
    t("Cassandra", Database, r"(?i)\bcassandra\b"),
    t("DynamoDB", Database, r"(?i)\bdynamo(?:db)?\b"),
    t("Elasticsearch", Database, r"(?i)\belastic ?search\b"),
    t("OpenSearch", Database, r"(?i)\bopen ?search\b"),
    t("CockroachDB", Database, r"(?i)\bcockroach(?:db)?\b"),
    t("Spanner", Database, r"\bSpanner\b"),
    t("Bigtable", Database, r"(?i)\bbig ?table\b"),
    t("BigQuery", DataPlatform, r"(?i)\bbig ?query\b"),
    t("Snowflake", DataPlatform, r"\bSnowflake\b"),
    t("Redshift", DataPlatform, r"(?i)\bredshift\b"),
    t("ClickHouse", Database, r"(?i)\bclick ?house\b"),
    t("HBase", Database, r"(?i)\bhbase\b"),
    t("Neo4j", Database, r"(?i)\bneo4j\b"),
    t("InfluxDB", Database, r"(?i)\binflux(?:db)?\b"),
    t("TimescaleDB", Database, r"(?i)\btimescale(?:db)?\b"),
    t("Firestore", Database, r"(?i)\bfirestore\b"),
    t("Firebase", Cloud, r"(?i)\bfirebase\b"),
    t("Supabase", Cloud, r"(?i)\bsupabase\b"),
    t("Couchbase", Database, r"(?i)\bcouchbase\b"),
    t("CouchDB", Database, r"(?i)\bcouch ?db\b"),
    t("Oracle Database", Database, r"\bOracle(?: DB| Database)?\b"),
    t("SQL Server", Database, r"(?i)\b(?:sql server|mssql)\b"),
    t("Aurora", Database, r"\bAurora\b"),
    t("RDS", Database, r"\bRDS\b"),
    t("ScyllaDB", Database, r"(?i)\bscylla(?:db)?\b"),
    t("etcd", Infra, r"(?i)\betcd\b"),
    t("ZooKeeper", Infra, r"(?i)\bzoo ?keeper\b"),
    t("Consul", Infra, r"\bConsul\b"),
    t(
        "HashiCorp Vault",
        Security,
        r"(?i:\bhashicorp vault\b)|\bVault\b",
    ),
    t("S3", Database, r"(?i)\bs3\b"),
    t("MinIO", Database, r"(?i)\bmin ?io\b"),
    t("HDFS", DataPlatform, r"(?i)\bhdfs\b"),
    t("Parquet", DataPlatform, r"(?i)\bparquet\b"),
    t("Avro", DataPlatform, r"(?i)\bavro\b"),
    t("Delta Lake", DataPlatform, r"(?i)\bdelta lake\b"),
    t(
        "Apache Iceberg",
        DataPlatform,
        r"(?i:\bapache iceberg\b)|\bIceberg tables?\b",
    ),
    t("pgvector", Database, r"(?i)\bpg ?vector\b"),
    t("Pinecone", Database, r"(?i)\bpinecone\b"),
    t("FAISS", Ml, r"(?i)\bfaiss\b"),
    t("Milvus", Database, r"(?i)\bmilvus\b"),
    t("Weaviate", Database, r"(?i)\bweaviate\b"),
    // Messaging & data processing
    t("Kafka", Queue, r"(?i)\bkafka\b"),
    t("RabbitMQ", Queue, r"(?i)\brabbit ?mq\b"),
    t("ActiveMQ", Queue, r"(?i)\bactive ?mq\b"),
    t("Pulsar", Queue, r"(?i)\bapache pulsar\b|\bPulsar\b"),
    t("NATS", Queue, r"\bNATS\b"),
    t("SQS", Queue, r"(?i)\bsqs\b"),
    t("SNS", Queue, r"\bSNS\b"),
    t("Kinesis", Queue, r"(?i)\bkinesis\b"),
    t("Pub/Sub", Queue, r"(?i)\bpub ?/ ?sub\b|\bpubsub\b"),
    t("Celery", Queue, r"(?i)\bcelery\b"),
    t("Sidekiq", Queue, r"(?i)\bsidekiq\b"),
    t("ZeroMQ", Queue, r"(?i)\b(?:zero ?mq|zmq)\b"),
    t("Flink", DataPlatform, r"(?i)\bflink\b"),
    t("Spark", DataPlatform, r"(?i)\b(?:apache )?spark\b"),
    t("Airflow", DataPlatform, r"(?i)\bairflow\b"),
    t("dbt", DataPlatform, r"\bdbt\b"),
    t("Debezium", DataPlatform, r"(?i)\bdebezium\b"),
    t("Apache Beam", DataPlatform, r"(?i)\bapache beam\b"),
    t("Hadoop", DataPlatform, r"(?i)\bhadoop\b"),
    t("Databricks", DataPlatform, r"(?i)\bdatabricks\b"),
    t("Temporal", Infra, r"\bTemporal\b"),
    // Cloud & infrastructure
    t("AWS", Cloud, r"(?i)\baws\b|\bamazon web services\b"),
    t("GCP", Cloud, r"(?i)\bgcp\b|\bgoogle cloud(?: platform)?\b"),
    t("Azure", Cloud, r"(?i)\bazure\b"),
    t("EC2", Cloud, r"(?i)\bec2\b"),
    t("AWS Lambda", Cloud, r"(?i:\baws lambda\b)|\bLambdas?\b"),
    t("ECS", Cloud, r"\bECS\b"),
    t("EKS", Cloud, r"\bEKS\b"),
    t("Fargate", Cloud, r"(?i)\bfargate\b"),
    t("CloudFront", Cloud, r"(?i)\bcloud ?front\b"),
    t("CloudWatch", Observability, r"(?i)\bcloud ?watch\b"),
    t("Route 53", Cloud, r"(?i)\broute ?53\b"),
    t("IAM", Security, r"\bIAM\b"),
    t("VPC", Cloud, r"\bVPCs?\b"),
    t("API Gateway", Cloud, r"(?i)\bapi gateways?\b"),
    t("Cloud Run", Cloud, r"(?i)\bcloud run\b"),
    t("GKE", Cloud, r"\bGKE\b"),
    t("Cloudflare", Cloud, r"(?i)\bcloud ?flare\b"),
    t("Fastly", Cloud, r"(?i)\bfastly\b"),
    t("Akamai", Cloud, r"(?i)\bakamai\b"),
    t("Vercel", Cloud, r"(?i)\bvercel\b"),
    t("Netlify", Cloud, r"(?i)\bnetlify\b"),
    t("Heroku", Cloud, r"(?i)\bheroku\b"),
    t("DigitalOcean", Cloud, r"(?i)\bdigital ?ocean\b"),
    t("Kubernetes", Infra, r"(?i)\bkubernetes\b|\bk8s\b"),
    t("Docker", Infra, r"(?i)\bdocker\b"),
    t("Helm", Infra, r"\bHelm\b"),
    t("Terraform", Infra, r"(?i)\bterraform\b"),
    t("Pulumi", Infra, r"(?i)\bpulumi\b"),
    t("Ansible", Infra, r"(?i)\bansible\b"),
    t("Istio", Infra, r"(?i)\bistio\b"),
    t("Envoy", Infra, r"\bEnvoy\b"),
    t("Linkerd", Infra, r"(?i)\blinkerd\b"),
    t("Nginx", Infra, r"(?i)\bnginx\b"),
    t("HAProxy", Infra, r"(?i)\bha ?proxy\b"),
    t("Traefik", Infra, r"(?i)\btraefik\b"),
    t("Varnish", Infra, r"\bVarnish\b"),
    t("CDN", Infra, r"(?i)\bcdns?\b"),
    t("Linux", Infra, r"(?i)\blinux\b"),
    t("macOS", Infra, r"(?i)\bmac ?os\b"),
    t("iOS", Infra, r"\biOS\b"),
    t("Android", Infra, r"(?i)\bandroid\b"),
    t("Nomad", Infra, r"\bNomad\b"),
    // CI/CD & developer tools
    t("Jenkins", Tool, r"(?i)\bjenkins\b"),
    t("GitHub Actions", Tool, r"(?i)\bgithub actions\b"),
    t("GitLab CI", Tool, r"(?i)\bgitlab ci\b"),
    t("CircleCI", Tool, r"(?i)\bcircle ?ci\b"),
    t("Argo CD", Tool, r"(?i)\bargo ?cd\b"),
    t("Spinnaker", Tool, r"(?i)\bspinnaker\b"),
    t("Git", Tool, r"(?i)\bgit\b"),
    t("GitHub", Tool, r"(?i)\bgithub\b"),
    t("GitLab", Tool, r"(?i)\bgitlab\b"),
    t("Bitbucket", Tool, r"(?i)\bbitbucket\b"),
    t("Jira", Tool, r"(?i)\bjira\b"),
    t("Confluence", Tool, r"\bConfluence\b"),
    t("Bazel", Tool, r"(?i)\bbazel\b"),
    t("Gradle", Tool, r"(?i)\bgradle\b"),
    t("Maven", Tool, r"\bMaven\b"),
    t("npm", Tool, r"(?i)\bnpm\b"),
    t("Webpack", Tool, r"(?i)\bwebpack\b"),
    t("Vite", Tool, r"\bVite\b"),
    // Observability
    t("Prometheus", Observability, r"(?i)\bprometheus\b"),
    t("Grafana", Observability, r"(?i)\bgrafana\b"),
    t("Datadog", Observability, r"(?i)\bdata ?dog\b"),
    t("New Relic", Observability, r"(?i)\bnew relic\b"),
    t("Splunk", Observability, r"(?i)\bsplunk\b"),
    t("ELK", Observability, r"\bELK\b"),
    t("Kibana", Observability, r"(?i)\bkibana\b"),
    t("Logstash", Observability, r"(?i)\blogstash\b"),
    t("Jaeger", Observability, r"(?i)\bjaeger\b"),
    t("Zipkin", Observability, r"(?i)\bzipkin\b"),
    t(
        "OpenTelemetry",
        Observability,
        r"(?i)\bopen ?telemetry\b|\botel\b",
    ),
    t("Sentry", Observability, r"(?i)\bsentry\b"),
    t("PagerDuty", Observability, r"(?i)\bpager ?duty\b"),
    t("Honeycomb", Observability, r"\bHoneycomb\b"),
    // Testing
    t("Jest", Testing, r"\bJest\b"),
    t("Cypress", Testing, r"(?i)\bcypress\b"),
    t("Playwright", Testing, r"(?i)\bplaywright\b"),
    t("Selenium", Testing, r"(?i)\bselenium\b"),
    t("JUnit", Testing, r"(?i)\bjunit\b"),
    t("pytest", Testing, r"(?i)\bpy ?test\b"),
    t("Postman", Testing, r"(?i)\bpostman\b"),
    t("JMeter", Testing, r"(?i)\bjmeter\b"),
    // Security & identity
    t("OAuth", Security, r"(?i)\boauth ?2?(?:\.0)?\b"),
    t("JWT", Security, r"(?i)\bjwts?\b"),
    t("SAML", Security, r"(?i)\bsaml\b"),
    t(
        "OpenID Connect",
        Security,
        r"(?i)\bopen ?id(?: connect)?\b|\boidc\b",
    ),
    t("LDAP", Security, r"(?i)\bldap\b"),
    t("mTLS", Security, r"(?i)\bmtls\b|\bmutual tls\b"),
    t("TLS", Security, r"(?i)\btls\b|\bssl\b"),
    t("Auth0", Security, r"(?i)\bauth0\b"),
    t("Okta", Security, r"(?i)\bokta\b"),
    t("Keycloak", Security, r"(?i)\bkeycloak\b"),
    // ML / AI
    t("TensorFlow", Ml, r"(?i)\btensor ?flow\b"),
    t("PyTorch", Ml, r"(?i)\bpy ?torch\b"),
    t("scikit-learn", Ml, r"(?i)\bscikit-?learn\b|\bsklearn\b"),
    t("pandas", Ml, r"(?i)\bpandas\b"),
    t("NumPy", Ml, r"(?i)\bnum ?py\b"),
    t("Jupyter", Ml, r"(?i)\bjupyter\b"),
    t("Hugging Face", Ml, r"(?i)\bhugging ?face\b"),
    t("OpenAI", Ml, r"(?i)\bopen ?ai\b"),
    t("LangChain", Ml, r"(?i)\blang ?chain\b"),
    t("CUDA", Ml, r"(?i)\bcuda\b"),
    // Protocols
    t("HTTP/2", Protocol, r"(?i)\bhttp ?/ ?2\b"),
    t("HTTP/3", Protocol, r"(?i)\bhttp ?/ ?3\b"),
    t("TCP", Protocol, r"(?i)\btcp\b"),
    t("UDP", Protocol, r"(?i)\budp\b"),
    t("DNS", Protocol, r"(?i)\bdns\b"),
    t("QUIC", Protocol, r"(?i)\bquic\b"),
    t("WebRTC", Protocol, r"(?i)\bweb ?rtc\b"),
    t("MQTT", Protocol, r"(?i)\bmqtt\b"),
    t("AMQP", Protocol, r"(?i)\bamqp\b"),
    // Product & business tools
    t("Stripe", Tool, r"\bStripe\b"),
    t("Twilio", Tool, r"(?i)\btwilio\b"),
    t("Mixpanel", Tool, r"(?i)\bmixpanel\b"),
    t("Amplitude", Tool, r"\bAmplitude\b"),
    t("LaunchDarkly", Tool, r"(?i)\blaunch ?darkly\b"),
    t("Optimizely", Tool, r"(?i)\boptimizely\b"),
    t("Tableau", Tool, r"(?i)\btableau\b"),
    t("Looker", Tool, r"\bLooker\b"),
    t("Metabase", Tool, r"(?i)\bmetabase\b"),
    t("Figma", Tool, r"(?i)\bfigma\b"),
    t("Salesforce", Tool, r"(?i)\bsalesforce\b"),
    t("Shopify", Tool, r"(?i)\bshopify\b"),
];

struct Compiled {
    set: RegexSet,
    regexes: Vec<Regex>,
}

static COMPILED: LazyLock<Compiled> = LazyLock::new(|| {
    let patterns: Vec<&str> = TECHS.iter().map(|t| t.pattern).collect();
    Compiled {
        set: RegexSet::new(&patterns).expect("technology lexicon patterns are valid"),
        regexes: patterns
            .iter()
            .map(|p| Regex::new(p).expect("technology lexicon pattern is valid"))
            .collect(),
    }
});

/// Technologies mentioned in `text`, ordered by first occurrence, unique.
pub(crate) fn find_technologies(text: &str) -> Vec<&'static Tech> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let compiled = &*COMPILED;
    let mut found: Vec<(usize, usize)> = compiled
        .set
        .matches(text)
        .into_iter()
        .filter_map(|i| compiled.regexes[i].find(text).map(|m| (m.start(), i)))
        .collect();
    found.sort();
    let mut out: Vec<&'static Tech> = Vec::new();
    for (_, i) in found {
        let tech = &TECHS[i];
        if !out.iter().any(|t| t.name == tech.name) {
            out.push(tech);
        }
    }
    out
}

/// Kind of a canonical technology name.
pub(crate) fn tech_kind(name: &str) -> Option<TechKind> {
    TECHS.iter().find(|t| t.name == name).map(|t| t.kind)
}

/// Number of lexicon entries (tests / docs).
#[cfg(test)]
pub(crate) fn lexicon_size() -> usize {
    TECHS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<&'static str> {
        find_technologies(text).iter().map(|t| t.name).collect()
    }

    #[test]
    fn lexicon_compiles_and_is_large() {
        assert!(lexicon_size() >= 150, "{}", lexicon_size());
        assert!(names("nothing technical here").is_empty());
    }

    #[test]
    fn canonical_casing_and_order() {
        assert_eq!(
            names("we moved from postgres and k8s to kafka on aws"),
            vec!["PostgreSQL", "Kubernetes", "Kafka", "AWS"]
        );
        assert_eq!(names("C++ and C# and .NET"), vec!["C++", "C#", ".NET"]);
        assert_eq!(
            names("built with node.js and Next.js"),
            vec!["Node.js", "Next.js"]
        );
    }

    #[test]
    fn ambiguous_words_need_context() {
        assert!(names("how did the team react to that").is_empty());
        assert!(names("let's go with that, rest assured").is_empty());
        assert_eq!(names("the service is written in Go"), vec!["Go"]);
        assert_eq!(names("our React frontend"), vec!["React"]);
        assert_eq!(names("a REST API"), vec!["REST"]);
    }
}
