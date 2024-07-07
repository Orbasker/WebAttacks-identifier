import pyshark
import pandas as pd
from scapy.all import *


def extract_features(pcap_file):
    # Open the pcap file with raw data included
    cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=True)

    features = []

    for packet in cap:
        # print("extracting features")
        try:
            # Convert PyShark packet to Scapy packet
            raw_packet = bytes.fromhex(packet.get_raw_packet().hex())
            scapy_pkt = Ether(raw_packet)

            # Initialize default feature values
            protocol_type = "Unknown"
            service = "Unknown"
            flag = "0"
            src_bytes = 0
            dst_bytes = 0
            land = 0
            wrong_fragment = 0
            urgent = 0
            hot = 0
            num_failed_logins = 0
            logged_in = 1
            num_compromised = 0
            root_shell = 0
            su_attempted = 0
            num_root = 0
            num_file_creations = 0
            num_shells = 0
            num_access_files = 0
            num_outbound_cmds = 0
            is_host_login = 0
            is_guest_login = 0
            count = 1
            srv_count = 1
            serror_rate = 0.0
            srv_serror_rate = 0.0
            rerror_rate = 0.0
            srv_rerror_rate = 0.0
            same_srv_rate = 0.0
            diff_srv_rate = 0.0
            srv_diff_host_rate = 0.0
            dst_host_count = 1
            dst_host_srv_count = 1
            dst_host_same_srv_rate = 0.0
            dst_host_diff_srv_rate = 0.0
            dst_host_same_src_port_rate = 0.0
            dst_host_srv_diff_host_rate = 0.0
            dst_host_serror_rate = 0.0
            dst_host_srv_serror_rate = 0.0
            dst_host_rerror_rate = 0.0
            dst_host_srv_rerror_rate = 0.0

            # Extract features using Scapy
            if IP in scapy_pkt:
                protocol_type = scapy_pkt[IP].proto
                flag = scapy_pkt[IP].flags
                src_bytes = len(scapy_pkt[IP].payload)
                dst_bytes = len(scapy_pkt[IP].payload)
                land = int(scapy_pkt[IP].src == scapy_pkt[IP].dst)
                wrong_fragment = int(scapy_pkt[IP].frag)
                # service = scapy_pkt[IP].dport

            if TCP in scapy_pkt:
                protocol_type = "TCP"
                service = "TCP"
                src_bytes = len(scapy_pkt[TCP].payload)
                dst_bytes = len(scapy_pkt[TCP].payload)
                urgent = scapy_pkt[TCP].urgptr
                hot = scapy_pkt[TCP].reserved
                num_failed_logins = scapy_pkt[TCP].window
                logged_in = scapy_pkt[TCP].ack
                num_compromised = scapy_pkt[TCP].seq
                root_shell = scapy_pkt[TCP].reserved
                su_attempted = scapy_pkt[TCP].reserved
                num_root = scapy_pkt[TCP].reserved
                num_file_creations = scapy_pkt[TCP].reserved
                num_shells = scapy_pkt[TCP].reserved
                num_access_files = scapy_pkt[TCP].reserved
                num_outbound_cmds = scapy_pkt[TCP].reserved
                is_host_login = scapy_pkt[TCP].reserved
                is_guest_login = scapy_pkt[TCP].reserved
                count = scapy_pkt[TCP].reserved
                srv_count = scapy_pkt[TCP].reserved
                serror_rate = scapy_pkt[TCP].reserved
                srv_serror_rate = scapy_pkt[TCP].reserved
                rerror_rate = scapy_pkt[TCP].reserved
                srv_rerror_rate = scapy_pkt[TCP].reserved
                same_srv_rate = scapy_pkt[TCP].reserved
                diff_srv_rate = scapy_pkt[TCP].reserved
                srv_diff_host_rate = scapy_pkt[TCP].reserved
                dst_host_count = scapy_pkt[TCP].reserved
                dst_host_srv_count = scapy_pkt[TCP].reserved
                dst_host_same_srv_rate = scapy_pkt[TCP].reserved
                dst_host_diff_srv_rate = scapy_pkt[TCP].reserved
                dst_host_same_src_port_rate = scapy_pkt[TCP].reserved
                dst_host_srv_diff_host_rate = scapy_pkt[TCP].reserved
                dst_host_serror_rate = scapy_pkt[TCP].reserved
                dst_host_srv_serror_rate = scapy_pkt[TCP].reserved
                dst_host_rerror_rate = scapy_pkt[TCP].reserved
                dst_host_srv_rerror_rate = scapy_pkt[TCP].reserved
                # service = scapy_pkt[TCP].dport

            if ARP in scapy_pkt:
                protocol_type = "ARP"
                service = "ARP"
                src_bytes = len(scapy_pkt[ARP].payload)
                dst_bytes = len(scapy_pkt[ARP].payload)
                flag = scapy_pkt[ARP].op

            if ICMP in scapy_pkt:
                protocol_type = "ICMP"
                service = "ICMP"
                src_bytes = len(scapy_pkt[ICMP].payload)
                dst_bytes = len(scapy_pkt[ICMP].payload)
                flag = scapy_pkt[ICMP].type

            if UDP in scapy_pkt:
                protocol_type = "UDP"
                service = "UDP"
                src_bytes = len(scapy_pkt[UDP].payload)
                dst_bytes = len(scapy_pkt[UDP].payload)
                flag = scapy_pkt[UDP].chksum

            if protocol_type == "unknown":
                continue

            features.append(
                [
                    0,  # duration (needs proper calculation based on packet timestamps)
                    protocol_type,
                    service,
                    flag,
                    src_bytes,
                    dst_bytes,
                    land,
                    wrong_fragment,
                    urgent,
                    hot,
                    num_failed_logins,
                    logged_in,
                    num_compromised,
                    root_shell,
                    su_attempted,
                    num_root,
                    num_file_creations,
                    num_shells,
                    num_access_files,
                    num_outbound_cmds,
                    is_host_login,
                    is_guest_login,
                    count,
                    srv_count,
                    serror_rate,
                    srv_serror_rate,
                    rerror_rate,
                    srv_rerror_rate,
                    same_srv_rate,
                    diff_srv_rate,
                    srv_diff_host_rate,
                    dst_host_count,
                    dst_host_srv_count,
                    dst_host_same_srv_rate,
                    dst_host_diff_srv_rate,
                    dst_host_same_src_port_rate,
                    dst_host_srv_diff_host_rate,
                    dst_host_serror_rate,
                    dst_host_srv_serror_rate,
                    dst_host_rerror_rate,
                    dst_host_srv_rerror_rate,
                ]
            )
        except AttributeError as e:
            # Skip packets that do not have the necessary fields
            print(e)
            continue

    # Create DataFrame from features
    df = pd.DataFrame(
        features,
        columns=[
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ],
    )

    return df


# Example usage
# df = extract_features("Teardrop Capture.pcap")
# print(df.head())
